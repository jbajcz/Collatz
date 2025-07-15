use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use std::fs;
use std::io::{self, Write, Read};

// Import serde for serialization/deserialization
use serde::{Serialize, Deserialize};
// Import bincode for efficient binary serialization
use bincode;

// Configuration constants
const BATCH_SIZE: usize = 100;
const WORKER_COUNT: usize = 3;
const SAVE_FILE_NAME: &str = "collatz_state.bin"; // File to save/load state
const TOTAL_RANGE_SIZE: u128 = 2000; // Total numbers to check in a single run, starting from min_found

#[derive(Debug)]
struct WorkerBatch {
    values_to_check: Vec<u128>,
    values_to_add: Vec<u128>,
    worker_id: usize,
}

#[derive(Debug)]
struct BatchResponse {
    worker_id: usize,
    found_results: Vec<bool>,
    current_min_found: u128, // The admin's current lowest unverified starting number
    success: bool,
}

// Derive Serialize and Deserialize to allow saving/loading this struct
#[derive(Debug, Serialize, Deserialize)]
struct CollatzChecker {
    main_array: Vec<u128>,
    // min_found represents the lowest starting number that has NOT yet been confirmed
    // to converge. All numbers from 5 up to min_found - 1 are considered confirmed.
    min_found: u128,
    max_stored: u128,
    // total_checked represents the count of starting numbers (from 5 upwards)
    // that have been fully verified to converge.
    total_checked: u128,
}

impl CollatzChecker {
    fn new() -> Self {
        let mut checker = CollatzChecker {
            main_array: Vec::new(),
            min_found: u128::MAX, // Temporarily set high, will be 5 after initial add_values
            max_stored: 0,
            total_checked: 0,
        };
        
        // Initialize with known values (1, 2, 4 loop)
        // These are added to main_array, but min_found is explicitly set to 5 afterwards.
        checker.add_values(&[1, 2, 4]);
        
        // Explicitly set min_found to the first number we actually need to check as a starting point.
        // All numbers < 5 (i.e., 1,2,3,4) are either handled (1,2,4) or will converge quickly (3 -> 10 -> 5 -> 16 -> ...).
        // 5 is the first number we will actively try to verify.
        checker.min_found = 5; 
        
        checker
    }
    
    // This function now handles adding values to the main_array, updating max_stored,
    // and performing the powers-of-2 extrapolation in a single, more efficient pass.
    fn add_values(&mut self, values: &[u128]) {
        // Collect all new values (direct and extrapolated) into a temporary vector.
        // Pre-allocate some capacity to reduce reallocations.
        let mut values_to_add_and_extrapolate = Vec::with_capacity(values.len() * 2);

        // Add the direct values from the worker batch
        values_to_add_and_extrapolate.extend_from_slice(values);

        // Generate extrapolated values (powers of 2) from the incoming direct values.
        // These incoming 'values' are the base candidates because they are newly discovered
        // parts of a converging sequence.
        for &base_value in values {
            let mut current_extrapolated = base_value;
            loop {
                // Check for overflow before multiplying by 2.
                // If `current_extrapolated` is already greater than half of `u128::MAX`,
                // then `current_extrapolated * 2` would overflow.
                if current_extrapolated > u128::MAX / 2 {
                    break; // Cannot multiply further without overflow
                }
                current_extrapolated *= 2;

                // Only add extrapolated values that are relevant (>= min_found).
                // We don't check `contains_value` here, as the final dedup will handle duplicates.
                if current_extrapolated >= self.min_found {
                    values_to_add_and_extrapolate.push(current_extrapolated);
                } else {
                    // If the extrapolated value is less than min_found, it would be purged anyway.
                    // Further multiples of 2 will also be less than min_found.
                    break; 
                }
            }
        }

        // Extend the main_array with all new and extrapolated values.
        self.main_array.extend(values_to_add_and_extrapolate);

        // Sort and deduplicate the entire main_array once.
        // This is the primary optimization: one sort/dedup instead of two.
        self.main_array.sort_unstable();
        self.main_array.dedup();

        // Update max_stored based on the (now sorted) last element, if the array is not empty.
        if let Some(&last_val) = self.main_array.last() {
            if last_val > self.max_stored {
                self.max_stored = last_val;
            }
        }

        // Always purge values below min_found to keep the array as small as possible.
        self.purge_below_min();
    }
    
    fn purge_below_min(&mut self) {
        // Retain only values that are greater than or equal to the current min_found.
        // Values below min_found are considered "solved" and no longer needed for lookups.
        self.main_array.retain(|&x| x >= self.min_found);
    }
    
    fn contains_value(&self, value: u128) -> bool {
        // If a value is less than min_found, it means it's already part of a sequence
        // that has been fully verified, so we consider it "found".
        if value < self.min_found {
            return true;
        }
        
        // Otherwise, perform a binary search in the main array.
        self.main_array.binary_search(&value).is_ok()
    }
    
    // Method to advance min_found and update total_checked.
    // This should be called by the admin thread after processing batches.
    fn advance_min_found(&mut self) {
        let initial_min_found = self.min_found;
        // Keep advancing min_found as long as the current min_found value
        // is present in the main_array. This means its sequence (or itself)
        // has been confirmed to reach a known cycle.
        while self.main_array.binary_search(&self.min_found).is_ok() {
            self.min_found += 1;
        }
        // The difference between the new min_found and the old one
        // represents the number of new starting values that have been confirmed.
        self.total_checked += (self.min_found - initial_min_found);
    }

    fn process_batch(&mut self, batch: &WorkerBatch) -> BatchResponse {
        let mut found_results = Vec::new();
        
        // Quick elimination: if all values in the batch are below the current min_found,
        // they are all considered found.
        let all_below_min = batch.values_to_check.iter().all(|&v| v < self.min_found);
        if all_below_min {
            found_results = vec![true; batch.values_to_check.len()];
        } else {
            // Quick elimination: if the minimum value in the batch is above the maximum
            // value stored in the main array, none of them can be found.
            let min_batch = batch.values_to_check.iter().min().copied().unwrap_or(u128::MAX);
            if min_batch > self.max_stored {
                found_results = vec![false; batch.values_to_check.len()];
            } else {
                // Perform individual searches for each value in the batch.
                for &value in &batch.values_to_check {
                    found_results.push(self.contains_value(value));
                }
            }
        }
        
        // Add new values (the sequence elements that were not found) to the main array.
        // This call will now also trigger the powers-of-2 extrapolation and single sort/dedup.
        if !batch.values_to_add.is_empty() {
            self.add_values(&batch.values_to_add);
        }
        
        // After potentially adding new values and performing extrapolation, try to advance min_found.
        // This is crucial for keeping min_found accurate and for purging.
        self.advance_min_found();

        BatchResponse {
            worker_id: batch.worker_id,
            found_results,
            current_min_found: self.min_found, // Send the updated min_found back to the worker
            success: true,
        }
    }
    
    fn get_stats(&self) -> String {
        format!(
            "Array size: {}, Min found: {}, Max stored: {}, Total checked (starting numbers): {}",
            self.main_array.len(),
            self.min_found,
            self.max_stored,
            self.total_checked
        )
    }
}

// Collatz sequence calculation
fn collatz_step(n: u128) -> Option<u128> {
    if n == 0 {
        return None; // Collatz sequence is for positive integers
    }
    
    if n % 2 == 0 {
        Some(n / 2)
    } else {
        // Check for overflow on 3n + 1
        // u128::MAX is 2^128 - 1. We need to ensure 3*n + 1 doesn't exceed this.
        // So, n must be less than (u128::MAX - 1) / 3.
        if n > (u128::MAX - 1) / 3 {
            None // Overflow would occur
        } else {
            Some(3 * n + 1)
        }
    }
}

// Worker thread function
fn worker_thread(
    worker_id: usize,
    start_value: u128,
    end_value: u128,
    batch_tx: mpsc::Sender<WorkerBatch>,
    response_rx: mpsc::Receiver<BatchResponse>,
) {
    println!("Worker {} starting range: {} to {}", worker_id, start_value, end_value);
    
    let mut current = start_value;
    let mut values_processed_in_range = 0; // Renamed for clarity, counts numbers in worker's assigned range
    let mut local_min_found = 5u128; // Local copy of the global min_found from admin
    
    while current <= end_value {
        let mut sequence = Vec::new();
        let mut temp_sequence = Vec::new();
        let mut n = current;
        
        // Quick check: if the current starting number is below the local min_found,
        // it means it has already been verified by the admin, so we skip it.
        if current < local_min_found {
            current += 1;
            values_processed_in_range += 1;
            continue;
        }
        
        // Generate Collatz sequence until we find a known value or overflow
        loop {
            // Add the current number to the temporary sequence.
            temp_sequence.push(n);
            
            // If the temporary sequence reaches BATCH_SIZE, send it to the admin for checking.
            if temp_sequence.len() >= BATCH_SIZE {
                let batch = WorkerBatch {
                    values_to_check: temp_sequence.clone(), // Clone to send, clear local copy
                    values_to_add: Vec::new(), // No values to add yet, just checking existence
                    worker_id,
                };
                
                // Send the batch to the admin. Handle potential send errors (e.g., admin thread died).
                if batch_tx.send(batch).is_err() {
                    println!("Worker {} batch send failed (admin channel closed)", worker_id);
                    return;
                }
                
                // Wait for the admin's response.
                match response_rx.recv() {
                    Ok(response) => {
                        // Update the worker's local min_found based on the admin's global state.
                        local_min_found = response.current_min_found;
                        
                        // Check if any values in the batch were found in the main array.
                        if let Some(found_index) = response.found_results.iter().position(|&x| x) {
                            // If a known value is found, add all preceding values in the temp_sequence
                            // (up to and including the found value) to the main sequence for this starting number.
                            sequence.extend(&temp_sequence[..=found_index]);
                            break; // Break the loop, this starting number's sequence is resolved.
                        }
                        // If no values were found in the batch, add all of them to the main sequence
                        // and clear temp_sequence to continue generating the current sequence.
                        sequence.extend(&temp_sequence);
                        temp_sequence.clear();
                    }
                    Err(_) => {
                        println!("Worker {} response receive failed (admin channel closed)", worker_id);
                        return;
                    }
                }
            }
            
            // Calculate the next Collatz step.
            match collatz_step(n) {
                Some(next_n) => n = next_n,
                None => {
                    // If an overflow occurs, print a message and break the sequence generation.
                    println!("Worker {} hit overflow at {}", worker_id, n);
                    break;
                }
            }
        }
        
        // After the sequence for 'current' is resolved (either found a known value or overflowed),
        // send the complete sequence (values_to_add) to the admin.
        if !sequence.is_empty() {
            let batch = WorkerBatch {
                values_to_check: Vec::new(), // No values to check, just adding
                values_to_add: sequence,
                worker_id,
            };
            
            if batch_tx.send(batch).is_err() {
                println!("Worker {} final batch send failed (admin channel closed)", worker_id);
                return;
            }
            
            // Wait for confirmation from admin and update local min_found.
            match response_rx.recv() {
                Ok(response) => {
                    local_min_found = response.current_min_found;
                }
                Err(_) => {
                    println!("Worker {} final response receive failed (admin channel closed)", worker_id);
                    return;
                }
            }
        }
        
        current += 1; // Move to the next starting number
        values_processed_in_range += 1;
        
        // Periodically report progress.
        if values_processed_in_range % 1000 == 0 {
            println!("Worker {} processed {} values in its range (local min_found: {})", 
                     worker_id, values_processed_in_range, local_min_found);
        }
    }
    
    println!("Worker {} completed! Processed {} values in its range", worker_id, values_processed_in_range);
}

// Admin thread function
// Now returns the final CollatzChecker state for saving
fn admin_thread(
    initial_checker: CollatzChecker, // Receive initial checker state
    batch_rx: mpsc::Receiver<WorkerBatch>,
    response_txs: Vec<mpsc::Sender<BatchResponse>>,
) -> CollatzChecker { // Return the final checker state
    let mut checker = initial_checker;
    let mut batches_processed = 0;
    let start_time = Instant::now();
    
    println!("Admin thread starting...");
    
    // Loop to receive batches from workers until the batch_tx channel is closed (meaning all workers finished).
    while let Ok(batch) = batch_rx.recv() {
        let response = checker.process_batch(&batch);
        
        // Send the response back to the correct worker based on its ID.
        if let Some(tx) = response_txs.get(batch.worker_id) {
            if tx.send(response).is_err() {
                println!("Admin failed to send response to worker {} (worker channel closed)", batch.worker_id);
            }
        }
        
        batches_processed += 1;
        
        // Periodically report progress and current stats.
        if batches_processed % 100 == 0 {
            let elapsed = start_time.elapsed();
            println!("Admin processed {} batches in {:?}", batches_processed, elapsed);
            println!("Stats: {}", checker.get_stats());
        }
    }
    
    println!("Admin thread finished. Final stats: {}", checker.get_stats());
    checker // Return the final checker state
}

// Function to save the CollatzChecker state to a file
fn save_state(checker: &CollatzChecker) -> io::Result<()> {
    let encoded: Vec<u8> = bincode::serialize(checker)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Bincode serialization error: {}", e)))?;
    
    let mut file = fs::File::create(SAVE_FILE_NAME)?;
    file.write_all(&encoded)?;
    println!("State saved to {}", SAVE_FILE_NAME);
    Ok(())
}

// Function to load the CollatzChecker state from a file
fn load_state() -> io::Result<CollatzChecker> {
    let mut file = fs::File::open(SAVE_FILE_NAME)?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)?;
    
    let decoded: CollatzChecker = bincode::deserialize(&encoded)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Bincode deserialization error: {}", e)))?;
    println!("State loaded from {}", SAVE_FILE_NAME);
    Ok(decoded)
}


fn main() {
    println!("Starting Collatz Conjecture Checker");
    println!("Configuration: {} workers, batch size {}", WORKER_COUNT, BATCH_SIZE);
    println!("Checking a total range of {} numbers per run.", TOTAL_RANGE_SIZE);
    
    // Try to load previous state, or create a new one
    let initial_checker_state = match load_state() {
        Ok(checker) => checker,
        Err(e) => {
            println!("Could not load state from {}: {}. Starting new computation.", SAVE_FILE_NAME, e);
            CollatzChecker::new()
        }
    };

    let start_checking_from = initial_checker_state.min_found;
    let range_per_worker = TOTAL_RANGE_SIZE / (WORKER_COUNT as u128);
    let remainder = TOTAL_RANGE_SIZE % (WORKER_COUNT as u128);

    // Create communication channels: one for workers to send batches to admin,
    // and one for each worker to receive responses from admin.
    let (batch_tx, batch_rx) = mpsc::channel();
    let mut response_txs = Vec::new();
    let mut worker_handles = Vec::new();
    
    // Spawn worker threads.
    for worker_id in 0..WORKER_COUNT {
        let (response_tx, response_rx) = mpsc::channel();
        response_txs.push(response_tx); // Store sender for admin to use
        
        let batch_tx_clone = batch_tx.clone(); // Clone sender for each worker
        
        // Define work ranges for each worker.
        // Each worker checks a contiguous block of starting numbers.
        let mut worker_start_value = start_checking_from + (worker_id as u128) * range_per_worker;
        let mut worker_end_value = worker_start_value + range_per_worker - 1;

        // Distribute the remainder of the range evenly among the first 'remainder' workers
        if (worker_id as u128) < remainder {
            worker_start_value += worker_id as u128;
            worker_end_value += (worker_id as u128) + 1;
        } else {
            worker_start_value += remainder;
            worker_end_value += remainder;
        }

        // Ensure end_value does not exceed u128::MAX
        worker_end_value = worker_end_value.min(u128::MAX);

        println!("Assigning Worker {} range: {} to {}", worker_id, worker_start_value, worker_end_value);
        
        let handle = thread::spawn(move || {
            worker_thread(worker_id, worker_start_value, worker_end_value, batch_tx_clone, response_rx);
        });
        
        worker_handles.push(handle);
    }
    
    // Start the admin thread, passing the initial checker state
    let admin_handle = thread::spawn(move || {
        admin_thread(initial_checker_state, batch_rx, response_txs)
    });
    
    // Wait for all worker threads to complete their assigned ranges.
    for (i, handle) in worker_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(_) => println!("Worker {} joined successfully", i),
            Err(_) => println!("Worker {} panicked", i),
        }
    }
    
    // Drop the main batch_tx sender. This signals to the admin_thread's batch_rx
    // that no more messages will be sent, allowing it to finish its loop.
    drop(batch_tx);
    
    // Wait for the admin thread to complete its final processing and get the final state.
    let final_checker_state = match admin_handle.join() {
        Ok(checker) => {
            println!("Admin thread joined successfully");
            checker
        },
        Err(_) => {
            println!("Admin thread panicked. State might not be consistent.");
            // In case of panic, we might want to load the last saved state or handle differently.
            // For now, we'll just create a new checker, but a more robust solution would be needed.
            CollatzChecker::new() 
        }
    };

    // Save the final state
    if let Err(e) = save_state(&final_checker_state) {
        eprintln!("Failed to save final state: {}", e);
    }
    
    println!("Collatz checker completed!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // Helper to clean up the save file before/after tests
    fn cleanup_test_file() {
        let _ = fs::remove_file(SAVE_FILE_NAME);
    }

    #[test]
    fn test_collatz_step() {
        assert_eq!(collatz_step(1), Some(4));
        assert_eq!(collatz_step(2), Some(1));
        assert_eq!(collatz_step(3), Some(10));
        assert_eq!(collatz_step(4), Some(2));
        assert_eq!(collatz_step(5), Some(16));
    }
    
    #[test]
    fn test_collatz_checker() {
        let mut checker = CollatzChecker::new();
        
        // Should contain initial values
        assert!(checker.contains_value(1));
        assert!(checker.contains_value(2));
        assert!(checker.contains_value(4));
        
        // Should not contain other values initially
        assert!(!checker.contains_value(3));
        assert!(!checker.contains_value(5));
        
        // Add some values
        checker.add_values(&[3, 5, 6]);
        assert!(checker.contains_value(3));
        assert!(checker.contains_value(5));
        assert!(checker.contains_value(6));
        
        // After adding values, min_found should still be 5 initially,
        // but advance_min_found should be able to move it.
        assert_eq!(checker.min_found, 5); // Still 5 because 5 hasn't been "confirmed" as a starting number yet.
        
        // Simulate a batch where 5's sequence is resolved (e.g., 5 -> 16 -> 8 -> 4).
        // If 5 is added to main_array, min_found should advance.
        let batch_for_5 = WorkerBatch {
            values_to_check: vec![],
            values_to_add: vec![5, 16, 8], // Assuming these are part of 5's sequence
            worker_id: 0,
        };
        checker.process_batch(&batch_for_5);
        // After processing batch for 5, and 5 is added to main_array,
        // min_found should advance past 5.
        // The advance_min_found function will be called internally by process_batch.
        assert!(checker.min_found > 5, "min_found should have advanced past 5");
        
        // Test purging
        let initial_len = checker.main_array.len();
        checker.min_found = 1000; // Manually set min_found high to force purging
        checker.purge_below_min();
        assert!(checker.main_array.len() < initial_len, "Array should have been purged");
    }

    #[test]
    fn test_save_load_state() {
        cleanup_test_file(); // Ensure no old file interferes

        let mut original_checker = CollatzChecker::new();
        original_checker.add_values(&[7, 10, 20]);
        original_checker.advance_min_found(); // Advance min_found after adding
        original_checker.min_found = 10; // Manually set for test
        original_checker.total_checked = 5;

        // Save the state
        assert!(save_state(&original_checker).is_ok());

        // Load the state
        let loaded_checker = load_state().expect("Failed to load state");

        // Verify loaded state matches original
        assert_eq!(loaded_checker.main_array, original_checker.main_array);
        assert_eq!(loaded_checker.min_found, original_checker.min_found);
        assert_eq!(loaded_checker.max_stored, original_checker.max_stored);
        assert_eq!(loaded_checker.total_checked, original_checker.total_checked);

        cleanup_test_file(); // Clean up after test
    }

    #[test]
    fn test_extrapolation() {
        let mut checker = CollatzChecker::new(); // Starts with [1, 2, 4], min_found = 5
        let initial_array_len = checker.main_array.len();
        let initial_max_stored = checker.max_stored;

        // Add a new value, say 3, which is known to converge (3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1)
        // Its sequence contains 5, 8, 10, 16, etc.
        // We expect powers of 2 for 3, 5, 8, 10, 16 etc. to be extrapolated.
        checker.add_values(&[3]); 
        
        // After adding 3, and its sequence (which includes 5, 8, 10, 16 etc. from CollatzChecker::new() and its path),
        // we expect multiples of 2 to be added.
        // For example, if 3 is added, 6, 12, 24... should be extrapolated.
        // If 5 is added (which it is, indirectly via 3's path), 10, 20, 40... should be extrapolated.
        
        // Assert that some extrapolated values are present
        assert!(checker.contains_value(6), "6 (3*2) should be extrapolated");
        assert!(checker.contains_value(12), "12 (6*2) should be extrapolated");
        assert!(checker.contains_value(20), "20 (10*2) should be extrapolated");
        assert!(checker.contains_value(32), "32 (16*2) should be extrapolated");
        assert!(checker.contains_value(40), "40 (20*2) should be extrapolated");

        // The array size should have increased significantly due to extrapolation
        assert!(checker.main_array.len() > initial_array_len, "Array size should increase after extrapolation");
        assert!(checker.max_stored > initial_max_stored, "Max stored should increase due to extrapolation");

        // Test with a value that is already a power of 2, like 8 (already in array)
        // Extrapolating 8 should add 16, 32, 64, etc.
        let old_len = checker.main_array.len();
        checker.add_values(&[8]); // 8 is already in the array from 3's sequence
        assert_eq!(checker.main_array.len(), old_len, "Adding 8 should not change length if already present");
        assert!(checker.contains_value(64), "64 (8*8) should be extrapolated if 8 was processed");
        assert!(checker.contains_value(128), "128 (64*2) should be extrapolated");
    }
}
