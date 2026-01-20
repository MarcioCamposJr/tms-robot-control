"""
Reproducibility Experiment System

This module implements an automated data collection system for testing 
robot positioning reproducibility. 

Features:
- Manual trial arming (user positions robot, then arms system)
- Automatic recording when coil reaches target
- 90 second data collection per trial
- Up to 20 trials with data export
- Robot ID included in filenames
- Target position in robot coordinate space
"""

import time
import csv
import json
import os
from datetime import datetime
from enum import Enum
import numpy as np


class ExperimentState(Enum):
    IDLE = "idle"
    ARMED = "armed"
    RECORDING = "recording"
    SAVING = "saving"


class ReproducibilityExperiment:
    """
    Manages reproducibility experiment trials with manual arming.
    
    Workflow:
    1. User arms the trial (arm_trial)
    2. System waits for "coil at target"
    3. Automatically records for duration seconds
    4. Saves data and returns to IDLE
    5. Repeat up to max_trials times
    """
    
    def __init__(self, robot_id, max_trials=20, duration=90.0, output_dir="data/reproducibility"):
        """
        Initialize reproducibility experiment.
        
        Args:
            robot_id (str): Robot identifier (e.g., "robot_1", "robot_2")
            max_trials (int): Maximum number of trials to run
            duration (float): Duration of each trial in seconds
            output_dir (str): Directory to save data files
        """
        self.robot_id = robot_id
        self.max_trials = max_trials
        self.duration = duration
        self.output_dir = output_dir
        
        # State management
        self.state = ExperimentState.IDLE
        self.current_trial = 0
        self.completed_trials = 0
        
        # Data collection
        self.trial_data = []
        self.start_time = None
        self.trial_start_timestamp = None
        
        # Results tracking
        self.trial_results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"ReproducibilityExperiment initialized for {robot_id}")
        print(f"Max trials: {max_trials}, Duration: {duration}s, Output: {output_dir}")
    
    def can_arm_trial(self):
        """Check if a new trial can be armed."""
        return (self.state == ExperimentState.IDLE and 
                self.completed_trials < self.max_trials)
    
    def arm_trial(self):
        """
        Arm the next trial. System will wait for "coil at target" to start recording.
        
        Returns:
            bool: True if armed successfully, False otherwise
        """
        if not self.can_arm_trial():
            print(f"Cannot arm trial: state={self.state.value}, completed={self.completed_trials}/{self.max_trials}")
            return False
        
        self.current_trial = self.completed_trials + 1
        self.state = ExperimentState.ARMED
        self.trial_data = []
        
        print(f"⚡ Trial {self.current_trial}/{self.max_trials} ARMED - Waiting for coil at target...")
        return True
    
    def is_armed(self):
        """Check if system is armed and waiting for target."""
        return self.state == ExperimentState.ARMED
    
    def is_recording(self):
        """Check if currently recording data."""
        return self.state == ExperimentState.RECORDING
    
    def start_recording(self, target_pos):
        """
        Start recording data when coil reaches target.
        
        Args:
            target_pos (list/array): Target position in robot coordinates [x, y, z, rx, ry, rz]
                                    This is captured once and remains fixed for the trial.
        """
        if self.state != ExperimentState.ARMED:
            print(f"Cannot start recording: state={self.state.value}")
            return False
        
        self.state = ExperimentState.RECORDING
        self.start_time = time.time()
        self.trial_start_timestamp = datetime.now()
        self.trial_data = []
        
        # Store target position (fixed for entire trial)
        self.target_pos = np.array(target_pos) if target_pos is not None else np.zeros(6)
        
        print(f"🔴 Trial {self.current_trial}/{self.max_trials} - RECORDING ({self.duration}s)...")
        print(f"   Target position: [{self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f}] mm")
        return True
    
    def update(self, coil_pos, timestamp):
        """
        Update trial with new coil position data.
        Target position is fixed and was captured at start_recording().
        
        Args:
            coil_pos (list/array): Current coil/robot position [x, y, z, rx, ry, rz]
            timestamp (float): Current timestamp
        
        Returns:
            bool: True if trial is complete, False otherwise
        """
        if self.state != ExperimentState.RECORDING:
            return False
        
        elapsed_time = timestamp - self.start_time
        
        # Use fixed target position
        coil_pos = np.array(coil_pos) if coil_pos is not None else np.zeros(6)
        
        # Calculate error
        error = self.target_pos - coil_pos
        error_xyz = error[:3]
        error_total = np.linalg.norm(error_xyz)
        
        # Store data point (target_pos is fixed, coil_pos changes)
        data_point = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'target_x': self.target_pos[0],
            'target_y': self.target_pos[1],
            'target_z': self.target_pos[2],
            'target_rx': self.target_pos[3] if len(self.target_pos) > 3 else 0,
            'target_ry': self.target_pos[4] if len(self.target_pos) > 4 else 0,
            'target_rz': self.target_pos[5] if len(self.target_pos) > 5 else 0,
            'coil_x': coil_pos[0],
            'coil_y': coil_pos[1],
            'coil_z': coil_pos[2],
            'coil_rx': coil_pos[3] if len(coil_pos) > 3 else 0,
            'coil_ry': coil_pos[4] if len(coil_pos) > 4 else 0,
            'coil_rz': coil_pos[5] if len(coil_pos) > 5 else 0,
            'error_x': error_xyz[0],
            'error_y': error_xyz[1],
            'error_z': error_xyz[2],
            'error_total': error_total
        }
        
        self.trial_data.append(data_point)
        
        # Check if duration exceeded
        if elapsed_time >= self.duration:
            return True
        
        return False
    
    def stop_recording(self):
        """Stop recording and save trial data."""
        if self.state != ExperimentState.RECORDING:
            print(f"Cannot stop recording: state={self.state.value}")
            return False
        
        self.state = ExperimentState.SAVING
        
        # Save trial data
        self._save_trial_data()
        
        # Update state
        self.completed_trials += 1
        self.state = ExperimentState.IDLE
        
        print(f"✅ Trial {self.current_trial}/{self.max_trials} complete! ({len(self.trial_data)} samples)")
        
        if self.completed_trials < self.max_trials:
            print(f"💡 Position robot for trial {self.completed_trials + 1} and send 'arm' command")
        else:
            print(f"🎉 All {self.max_trials} trials completed! Use 'export' to generate summary.")
        
        return True
    
    def cancel_current_trial(self):
        """Cancel current trial without saving."""
        if self.state in [ExperimentState.ARMED, ExperimentState.RECORDING]:
            self.state = ExperimentState.IDLE
            self.trial_data = []
            print(f"🚫 Trial {self.current_trial} cancelled")
            return True
        return False
    
    def get_trial_number(self):
        """Get current trial number."""
        return self.current_trial
    
    def get_elapsed_time(self):
        """Get elapsed time in current recording."""
        if self.state == ExperimentState.RECORDING and self.start_time:
            return time.time() - self.start_time
        return 0
    
    def get_status(self):
        """
        Get current experiment status.
        
        Returns:
            dict: Status information
        """
        status = {
            'robot_id': self.robot_id,
            'state': self.state.value,
            'current_trial': self.current_trial,
            'completed_trials': self.completed_trials,
            'max_trials': self.max_trials,
            'duration': self.duration,
            'samples_collected': len(self.trial_data) if self.trial_data else 0
        }
        
        if self.state == ExperimentState.RECORDING:
            status['elapsed_time'] = self.get_elapsed_time()
            status['remaining_time'] = max(0, self.duration - status['elapsed_time'])
        
        return status
    
    def _save_trial_data(self):
        """Save current trial data to CSV file."""
        if not self.trial_data:
            print("Warning: No data to save")
            return
        
        # Generate filename with robot_id
        timestamp_str = self.trial_start_timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.robot_id}_trial_{self.current_trial:02d}_{timestamp_str}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trial_data[0].keys())
            writer.writeheader()
            writer.writerows(self.trial_data)
        
        # Calculate statistics
        errors = np.array([d['error_total'] for d in self.trial_data])
        errors_xyz = np.array([[d['error_x'], d['error_y'], d['error_z']] for d in self.trial_data])
        
        trial_result = {
            'trial_number': self.current_trial,
            'robot_id': self.robot_id,
            'start_time': self.trial_start_timestamp.isoformat(),
            'duration': self.duration,
            'samples_collected': len(self.trial_data),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'mean_error_xyz': {
                'x': float(np.mean(errors_xyz[:, 0])),
                'y': float(np.mean(errors_xyz[:, 1])),
                'z': float(np.mean(errors_xyz[:, 2]))
            },
            'std_error_xyz': {
                'x': float(np.std(errors_xyz[:, 0])),
                'y': float(np.std(errors_xyz[:, 1])),
                'z': float(np.std(errors_xyz[:, 2]))
            },
            'file': filename
        }
        
        self.trial_results.append(trial_result)
        
        print(f"📁 Data saved to: {filename}")
        print(f"   Mean error: {trial_result['mean_error']:.3f}mm, Std: {trial_result['std_error']:.3f}mm")
    
    def export_summary(self):
        """Export summary of all trials to JSON."""
        if not self.trial_results:
            print("No trials to export")
            return None
        
        # Calculate overall statistics
        all_mean_errors = [t['mean_error'] for t in self.trial_results]
        all_std_errors = [t['std_error'] for t in self.trial_results]
        
        summary = {
            'robot_id': self.robot_id,
            'experiment_date': datetime.now().isoformat(),
            'total_trials': len(self.trial_results),
            'duration_per_trial': self.duration,
            'trials': self.trial_results,
            'overall_statistics': {
                'mean_error_across_trials': float(np.mean(all_mean_errors)),
                'std_error_across_trials': float(np.std(all_mean_errors)),
                'max_error_observed': float(max([t['max_error'] for t in self.trial_results])),
                'min_error_observed': float(min([t['min_error'] for t in self.trial_results]))
            }
        }
        
        # Save summary with robot_id
        filename = f"{self.robot_id}_reproducibility_summary.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary exported to: {filename}")
        print(f"   Overall mean error: {summary['overall_statistics']['mean_error_across_trials']:.3f}mm")
        
        return filepath
