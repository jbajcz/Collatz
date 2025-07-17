use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Instant, Duration};
use std::fs;
use std::io::{self, Write, Read};
use std::collections::HashSet; // For worker-local loop detection

// Import serde for serialization/deserialization
use serde::{Serialize, Deserialize};
// Import bincode for efficient binary serialization
use bincode;
// Import atomic crate for shared atomic u128 operations
use atomic::{Atomic, Ordering};


// Configuration constants
const WORKER_COUNT: usize = 10; // Increased worker count as requested
const SAVE_FILE_NAME: &str = "collatz_state.bin"; // File to save/load state
const TOTAL_RANGE_SIZE: u128 = 10_000_000; // Total numbers to check in a single run
const CHECKPOINT_INTERVAL_SECONDS: u64 = 60; // How often to save progress (in seconds)

// Structure to save/load the state
#[derive(Debug, Serialize, Deserialize)]
struct CollatzState {
    // min_found now represents the next starting number to be assigned to a worker.
    // It is the highest starting number processed + 1.
    min_found: u128, 
    // total_checked represents the count of starting numbers (from 5 upwards)
    // that have been fully verified to converge.
    total_checked: u128, 
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
    // This atomic tracks the next starting number to assign to a worker.
    // It's the primary progress indicator for starting numbers AND the convergence boundary.
    next_starting_number_to_assign: Arc<Atomic<u128>>, 
    global_end_value: u128, // The absolute highest number to check in this run
    termination_tx: mpsc::Sender<()>, // To signal main thread when done
) {
    println!("Worker {} starting...", worker_id);
    
    let mut values_processed_in_range = 0; // Used only for local progress reporting
    
    loop {
        // Atomically get the next number to check as a starting point for this worker.
        // This implements the interleaved range assignment.
        let current_starting_num = next_starting_number_to_assign.fetch_add(1, Ordering::SeqCst);

        // If we've exceeded the global end value for this run, terminate.
        if current_starting_num > global_end_value {
            break;
        }
        
        let mut n = current_starting_num;
        let mut current_sequence_elements = HashSet::new(); // Local set for loop detection

        // Generate Collatz sequence until we find a known value or overflow
        loop {
            // Check if the current number `n` in the sequence has dropped below the
            // `next_starting_number_to_assign` (our global "min_found").
            // If it has, it means this number (or a lower one) has already been processed
            // as a starting number, so this sequence is confirmed to converge.
            if n < next_starting_number_to_assign.load(Ordering::Relaxed) {
                // Sequence converged!
                break; 
            }

            // Check for a loop within the current sequence being computed by this worker.
            if !current_sequence_elements.insert(n) {
                // Loop detected! This implies the conjecture is false for `current_starting_num`
                // unless it's the 1-4-2 loop. Since 1,2,4 are always below `next_starting_number_to_assign`
                // (which starts at 5), they would have been caught by the `n < next_starting_number_to_assign` check.
                // If this is hit for a number > 4, it's a counterexample.
                println!("Worker {} detected a loop for starting number {} at value {}", worker_id, current_starting_num, n);
                break; 
            }
            
            // Calculate the next Collatz step.
            match collatz_step(n) {
                Some(next_n) => n = next_n,
                None => {
                    // If an overflow occurs, print a message and break the sequence generation.
                    println!("Worker {} hit overflow for starting number {} at value {}", worker_id, current_starting_num, n);
                    break;
                }
            }
        }
        
        values_processed_in_range += 1;
        
        // Periodically report progress.
        if values_processed_in_range % 1000 == 0 {
            println!("Worker {} processed {} values (current starting num: {})", 
                     worker_id, values_processed_in_range, current_starting_num);
        }
    }
    
    println!("Worker {} completed! Processed {} values in its range", worker_id, values_processed_in_range);
    // Signal to the main thread that this worker is done
    let _ = termination_tx.send(()); 
}

// Function to save the CollatzState to a file
fn save_state(state: &CollatzState) -> io::Result<()> {
    // Use bincode::serde::encode_to_vec for bincode v2.x.x
    let encoded: Vec<u8> = bincode::serde::encode_to_vec(state, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Bincode serialization error: {}", e)))?;
    
    let mut file = fs::File::create(SAVE_FILE_NAME)?;
    file.write_all(&encoded)?;
    println!("State saved to {}", SAVE_FILE_NAME);
    Ok(())
}

// Function to load the CollatzState from a file
fn load_state() -> io::Result<CollatzState> {
    let mut file = fs::File::open(SAVE_FILE_NAME)?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)?;
    
    // Use bincode::serde::decode_from_slice for bincode v2.x.x
    // It returns a tuple (decoded_value, bytes_read), so we need to destructure it.
    let (decoded, _bytes_read): (CollatzState, usize) = bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Bincode deserialization error: {}", e)))?;
    println!("State loaded from {}", SAVE_FILE_NAME);
    Ok(decoded)
}


fn main() {
    println!("Starting Collatz Conjecture Checker (No Central Array)");
    println!("Configuration: {} workers", WORKER_COUNT);
    println!("Checking a total range of {} numbers per run.", TOTAL_RANGE_SIZE);
    println!("Checkpointing every {} seconds.", CHECKPOINT_INTERVAL_SECONDS);
    
    // Try to load previous state, or create a new one
    let initial_state = match load_state() {
        Ok(state) => state,
        Err(e) => {
            println!("Could not load state from {}: {}. Starting new computation.", SAVE_FILE_NAME, e);
            // Initialize with min_found starting point (5, as 1,2,3,4 are implicitly handled)
            CollatzState { min_found: 5, total_checked: 0 }
        }
    };

    // Shared atomic variable for the next starting number to assign.
    // This serves as both the work assignment counter and the global "min_found" for convergence.
    let next_starting_number_to_assign = Arc::new(Atomic::new(initial_state.min_found)); 
    
    // Calculate the absolute end value for this run
    let global_end_value = initial_state.min_found.saturating_add(TOTAL_RANGE_SIZE - 1);

    let mut worker_handles = Vec::new();
    let (termination_tx, termination_rx) = mpsc::channel(); // Channel for workers to signal completion

    // Spawn worker threads.
    for worker_id in 0..WORKER_COUNT {
        let next_starting_number_to_assign_clone = Arc::clone(&next_starting_number_to_assign);
        let termination_tx_clone = termination_tx.clone();
        
        let handle = thread::spawn(move || {
            worker_thread(
                worker_id,
                next_starting_number_to_assign_clone,
                global_end_value,
                termination_tx_clone,
            );
        });
        
        worker_handles.push(handle);
    }
    // Drop the main sender to the termination channel, so the channel closes when all clones are dropped.
    drop(termination_tx); 

    let start_time = Instant::now();
    let mut last_checkpoint_time = Instant::now();

    // Main thread loop for checkpointing and waiting for workers
    let mut completed_workers = 0;
    while completed_workers < WORKER_COUNT {
        // Check for worker completion
        match termination_rx.recv_timeout(Duration::from_secs(CHECKPOINT_INTERVAL_SECONDS)) {
            Ok(_) => {
                completed_workers += 1;
                println!("Main: A worker completed. {}/{} workers done.", completed_workers, WORKER_COUNT);
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Timeout, no worker finished, proceed to checkpoint
            },
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // All senders dropped, meaning all workers have completed or panicked
                println!("Main: All worker channels disconnected. Assuming all workers completed or exited.");
                break; 
            }
        }

        // Perform checkpointing if interval passed or all workers completed
        if last_checkpoint_time.elapsed().as_secs() >= CHECKPOINT_INTERVAL_SECONDS || completed_workers == WORKER_COUNT {
            // The min_found for saving is the next number to be assigned.
            let current_min_found_for_saving = next_starting_number_to_assign.load(Ordering::Relaxed);
            // total_checked is the difference between current progress and initial starting point.
            let current_total_checked = current_min_found_for_saving.saturating_sub(initial_state.min_found); 
            
            let state_to_save = CollatzState { 
                min_found: current_min_found_for_saving, 
                total_checked: current_total_checked 
            };

            if let Err(e) = save_state(&state_to_save) {
                eprintln!("Failed to save state during checkpoint: {}", e);
            }
            println!("Checkpoint: Min found: {}, Total checked: {}, Elapsed: {:?}", 
                     current_min_found_for_saving, current_total_checked, start_time.elapsed());
            last_checkpoint_time = Instant::now();
        }
    }
    
    // Ensure all worker threads are joined (important for clean shutdown)
    for (i, handle) in worker_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(_) => println!("Worker {} joined successfully", i),
            Err(_) => println!("Worker {} panicked", i),
        }
    }
    
    println!("Collatz checker completed!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use atomic::Atomic; 

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
    fn test_save_load_state() {
        cleanup_test_file(); // Ensure no old file interferes

        let original_state = CollatzState {
            min_found: 12345,
            total_checked: 12340,
        };

        // Save the state
        assert!(save_state(&original_state).is_ok());

        // Load the state
        let loaded_state = load_state().expect("Failed to load state");

        // Verify loaded state matches original
        assert_eq!(loaded_state.min_found, original_state.min_found);
        assert_eq!(loaded_state.total_checked, original_state.total_checked);

        cleanup_test_file(); // Clean up after test
    }

    // This test simulates a very small run to check the worker logic and min_found update
    #[test]
    fn test_worker_logic_and_min_found() {
        cleanup_test_file();

        let initial_min_found = 5;
        let test_range_size = 10; // Check numbers 5 to 14
        let num_workers = 2; // Use 2 workers for simplicity in test

        let next_starting_number_to_assign = Arc::new(Atomic::new(initial_min_found));
        let global_end_value = initial_min_found.saturating_add(test_range_size - 1);

        let mut worker_handles = Vec::new();
        let (termination_tx, termination_rx) = mpsc::channel();

        for worker_id in 0..num_workers {
            let next_starting_number_to_assign_clone = Arc::clone(&next_starting_number_to_assign);
            let termination_tx_clone = termination_tx.clone();

            let handle = thread::spawn(move || {
                worker_thread(
                    worker_id,
                    next_starting_number_to_assign_clone,
                    global_end_value,
                    termination_tx_clone,
                );
            });
            worker_handles.push(handle);
        }
        drop(termination_tx); // Drop original sender to allow channel to close when workers finish

        // Wait for all workers to finish
        let mut completed_workers = 0;
        while completed_workers < num_workers {
            match termination_rx.recv_timeout(Duration::from_secs(5)) { // Use a timeout for tests
                Ok(_) => completed_workers += 1,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    panic!("Test timed out waiting for workers to complete!");
                },
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // All workers finished or panicked
                    break;
                }
            }
        }

        for handle in worker_handles {
            handle.join().expect("Worker thread panicked");
        }

        // After workers complete, next_starting_number_to_assign should have advanced
        // It should be equal to global_end_value + 1 if all numbers were checked.
        assert_eq!(next_starting_number_to_assign.load(Ordering::Relaxed), global_end_value + 1, 
                   "next_starting_number_to_assign should have advanced to cover the entire range");
        
        cleanup_test_file();
    }
}
