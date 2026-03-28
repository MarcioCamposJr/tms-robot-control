"""
Reproducibility Experiment System

This module implements a data collection system for testing 
robot positioning reproducibility. 

Features:
- Recording starts when user sends "go" (TRACK_TARGET objective)
- Recording stops manually when user sends "stop"
- Unlimited trials with sequential indexing
- On export, user provides a base filename; each trial gets an index
- Robot ID included in filenames
- Target position tracked dynamically
"""

import time
import csv
import json
import os
from datetime import datetime
import numpy as np


class ExperimentState:
    IDLE = "idle"
    RECORDING = "recording"
    SAVING = "saving"


class ReproducibilityExperiment:
    """
    Manages reproducibility experiment trials.
    
    Workflow:
    1. User sends "go" (TRACK_TARGET) → recording starts
    2. Data is collected each frame
    3. User sends "stop" → recording stops and data is saved
    4. Repeat as many times as needed
    5. User sends "export" with a filename → all trials exported
    """
    
    def __init__(self, robot_id, output_dir="data/reproducibility"):
        """
        Initialize reproducibility experiment.
        
        Args:
            robot_id (str): Robot identifier (e.g., "robot_1", "robot_2")
            output_dir (str): Directory to save data files
        """
        self.robot_id = robot_id
        self.output_dir = output_dir
        
        # State management
        self.state = ExperimentState.IDLE
        self.current_trial = 0
        
        # Data collection
        self.trial_data = []
        self.start_time = None
        self.trial_start_timestamp = None
        
        # Results tracking
        self.trial_results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"ReproducibilityExperiment initialized for {robot_id}")
        print(f"Output: {output_dir}")
    
    def is_recording(self):
        """Check if currently recording data."""
        return self.state == ExperimentState.RECORDING
    
    def start_recording(self, target_pos=None):
        """
        Start recording data for a new trial.
        Called when user sends "go" (TRACK_TARGET objective).
        
        Args:
            target_pos (list/array, optional): Initial target position in robot coordinates [x, y, z, rx, ry, rz]
        """
        if self.state == ExperimentState.RECORDING:
            print("Already recording, ignoring start_recording call")
            return False
        
        self.current_trial += 1
        self.state = ExperimentState.RECORDING
        self.start_time = time.time()
        self.trial_start_timestamp = datetime.now()
        self.trial_data = []
        
        # Reset head velocity tracking
        self._prev_head_pose = None
        self._prev_timestamp = None
        
        print(f"🔴 Trial {self.current_trial} - RECORDING started...")
        if target_pos is not None:
            target_pos = np.array(target_pos)
            print(f"   Initial target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] mm")
        return True
    
    def update(self, coil_pos, raw_displacement, timestamp, head_pose=None, target_from_displacement=None, target_from_head=None, distance_coils=None, repulsion_intensity=0.0, repulsion_zone="NONE"):
        """
        Update trial with new data.
        
        The primary error metric is raw_displacement — the vector from the coil to the target
        as computed by neuronavigation, BEFORE PID processing. Its norm is the positioning error.
        
        Head pose is recorded for computing head movement velocity between frames.
        
        Args:
            coil_pos (list/array): Current robot TCP pose [x, y, z, rx, ry, rz] (from robot encoders)
            raw_displacement (list/array): Displacement from coil to target [dx, dy, dz, drx, dry, drz]
                                           (before PID - THIS is the actual error)
            timestamp (float): Current timestamp
            head_pose (list/array): Head pose in robot space [x, y, z, rx, ry, rz] (for velocity calculation)
            target_from_displacement (list/array): Target estimated as robot_pose + displacement
            target_from_head (list/array): Target estimated from head tracker + calibration
            distance_coils (float): Distance between the two robots/coils in mm
            repulsion_intensity (float): Current repulsion brake magnitude
            repulsion_zone (str): Current repulsion zone ("NONE", "APPROACH", or "WORKING")
        """
        if self.state != ExperimentState.RECORDING:
            return
        
        elapsed_time = timestamp - self.start_time
        
        # Convert to numpy arrays
        coil = np.array(coil_pos) if coil_pos is not None else np.zeros(6)
        disp = np.array(raw_displacement) if raw_displacement is not None else np.zeros(6)
        head = np.array(head_pose) if head_pose is not None else np.zeros(6)
        tgt_disp = np.array(target_from_displacement) if target_from_displacement is not None else np.zeros(6)
        tgt_head = np.array(target_from_head) if target_from_head is not None else np.zeros(6)
        
        # The displacement IS the error (vector from coil to target)
        error_xyz = disp[:3]
        error_total = np.linalg.norm(error_xyz)
        
        # Calculate head velocity (mm/s) from consecutive samples
        head_velocity = 0.0
        head_velocity_xyz = np.zeros(3)
        if hasattr(self, '_prev_head_pose') and hasattr(self, '_prev_timestamp') and self._prev_head_pose is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 0:
                head_delta = head[:3] - self._prev_head_pose[:3]
                head_velocity_xyz = head_delta / dt
                head_velocity = np.linalg.norm(head_velocity_xyz)
        
        # Store for next frame
        self._prev_head_pose = head.copy() if head_pose is not None else None
        self._prev_timestamp = timestamp
        
        # Store data point
        data_point = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            # Error = raw displacement (coil → target, before PID)
            'error_x': disp[0],
            'error_y': disp[1],
            'error_z': disp[2],
            'error_rx': disp[3] if len(disp) > 3 else 0,
            'error_ry': disp[4] if len(disp) > 4 else 0,
            'error_rz': disp[5] if len(disp) > 5 else 0,
            'error_total': error_total,
            # Robot TCP pose (from encoders)
            'coil_x': coil[0],
            'coil_y': coil[1],
            'coil_z': coil[2],
            'coil_rx': coil[3] if len(coil) > 3 else 0,
            'coil_ry': coil[4] if len(coil) > 4 else 0,
            'coil_rz': coil[5] if len(coil) > 5 else 0,
            # Head pose in robot space
            'head_x': head[0],
            'head_y': head[1],
            'head_z': head[2],
            'head_rx': head[3] if len(head) > 3 else 0,
            'head_ry': head[4] if len(head) > 4 else 0,
            'head_rz': head[5] if len(head) > 5 else 0,
            # Head velocity (mm/s)
            'head_vel_x': head_velocity_xyz[0],
            'head_vel_y': head_velocity_xyz[1],
            'head_vel_z': head_velocity_xyz[2],
            'head_vel_total': head_velocity,
            # Target from displacement method
            'target_disp_x': tgt_disp[0],
            'target_disp_y': tgt_disp[1],
            'target_disp_z': tgt_disp[2],
            'target_disp_rx': tgt_disp[3] if len(tgt_disp) > 3 else 0,
            'target_disp_ry': tgt_disp[4] if len(tgt_disp) > 4 else 0,
            'target_disp_rz': tgt_disp[5] if len(tgt_disp) > 5 else 0,
            # Target from head tracker method
            'target_head_x': tgt_head[0],
            'target_head_y': tgt_head[1],
            'target_head_z': tgt_head[2],
            'target_head_rx': tgt_head[3] if len(tgt_head) > 3 else 0,
            'target_head_ry': tgt_head[4] if len(tgt_head) > 4 else 0,
            'target_head_rz': tgt_head[5] if len(tgt_head) > 5 else 0,
            # Repulsion data
            'distance_coils': distance_coils if distance_coils is not None else -1,
            'repulsion_intensity': repulsion_intensity,
            'repulsion_zone': repulsion_zone
        }
        
        self.trial_data.append(data_point)

    
    def stop_recording(self):
        """Stop recording and save trial data."""
        if self.state != ExperimentState.RECORDING:
            print(f"Cannot stop recording: state={self.state}")
            return False
        
        self.state = ExperimentState.SAVING
        
        # Save trial data
        self._save_trial_data()
        
        # Update state
        self.state = ExperimentState.IDLE
        
        print(f"✅ Trial {self.current_trial} complete! ({len(self.trial_data)} samples)")
        
        return True
    
    def cancel_current_trial(self):
        """Cancel current trial without saving."""
        if self.state == ExperimentState.RECORDING:
            self.state = ExperimentState.IDLE
            self.trial_data = []
            self.current_trial -= 1  # Don't count cancelled trial
            print(f"🚫 Trial cancelled")
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
            'state': self.state,
            'current_trial': self.current_trial,
            'completed_trials': len(self.trial_results),
            'samples_collected': len(self.trial_data) if self.trial_data else 0
        }
        
        if self.state == ExperimentState.RECORDING:
            status['elapsed_time'] = self.get_elapsed_time()
        
        return status
    
    def _save_trial_data(self):
        """Save current trial data to CSV file."""
        if not self.trial_data:
            print("Warning: No data to save")
            return
        
        # Generate filename with robot_id and trial index
        timestamp_str = self.trial_start_timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.robot_id}_trial_{self.current_trial:02d}_{timestamp_str}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trial_data[0].keys())
            writer.writeheader()
            writer.writerows(self.trial_data)
        
        # Calculate statistics (error = raw displacement before PID)
        errors = np.array([d['error_total'] for d in self.trial_data])
        errors_xyz = np.array([[d['error_x'], d['error_y'], d['error_z']] for d in self.trial_data])
        
        # Head velocity statistics
        head_velocities = np.array([d['head_vel_total'] for d in self.trial_data])
        
        duration = self.trial_data[-1]['elapsed_time'] if self.trial_data else 0
        
        trial_result = {
            'trial_number': self.current_trial,
            'robot_id': self.robot_id,
            'start_time': self.trial_start_timestamp.isoformat(),
            'duration': duration,
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
            'mean_head_velocity': float(np.mean(head_velocities)),
            'max_head_velocity': float(np.max(head_velocities)),
            'file': filename
        }
        
        self.trial_results.append(trial_result)
        
        print(f"📁 Data saved to: {filename}")
        print(f"   Error (displacement): mean={trial_result['mean_error']:.3f}mm, std={trial_result['std_error']:.3f}mm")
        print(f"   Head velocity: mean={trial_result['mean_head_velocity']:.1f}mm/s, max={trial_result['max_head_velocity']:.1f}mm/s")
    
    def export_summary(self, base_filename=None):
        """
        Export summary of all trials to JSON.
        
        Args:
            base_filename (str, optional): Base name for the export file.
                                           If provided, files are named: {base_filename}_trial_01.csv, etc.
                                           Summary is named: {base_filename}_summary.json
                                           If not provided, uses robot_id as base.
        
        Returns:
            str: Path to the summary file, or None if no trials.
        """
        if not self.trial_results:
            print("No trials to export")
            return None
        
        # Use provided name or default to robot_id
        name = base_filename if base_filename else f"{self.robot_id}_reproducibility"
        
        # Calculate overall statistics
        all_mean_errors = [t['mean_error'] for t in self.trial_results]
        
        summary = {
            'robot_id': self.robot_id,
            'experiment_date': datetime.now().isoformat(),
            'base_filename': name,
            'total_trials': len(self.trial_results),
            'trials': self.trial_results,
            'overall_statistics': {
                'mean_error_across_trials': float(np.mean(all_mean_errors)),
                'std_error_across_trials': float(np.std(all_mean_errors)),
                'max_error_observed': float(max([t['max_error'] for t in self.trial_results])),
                'min_error_observed': float(min([t['min_error'] for t in self.trial_results]))
            }
        }
        
        # Save summary
        filename = f"{name}_summary.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary exported to: {filename}")
        print(f"   Total trials: {len(self.trial_results)}")
        print(f"   Overall mean error: {summary['overall_statistics']['mean_error_across_trials']:.3f}mm")
        
        return filepath
