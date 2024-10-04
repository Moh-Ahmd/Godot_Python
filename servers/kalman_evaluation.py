import os
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class KalmanFilterEvaluator:
    """
    A class for evaluating and visualizing the performance of Kalman filters.

    This class provides methods to collect data points,
    evaluate filter performance, and generate plots to visualize
    the results of Kalman filtering.
    """

    def __init__(self, filter_name: str):
        """
        Initialize the KalmanFilterEvaluator.

        Args:
            filter_name (str): The name of the Kalman filter being evaluated.
        """
        self.filter_name = filter_name
        self.true_states: List[np.ndarray] = []
        self.measurements: List[np.ndarray] = []
        self.estimated_states: List[np.ndarray] = []
        self.predicted_states: List[np.ndarray] = []
        self.ensure_output_directory()

    def ensure_output_directory(self) -> None:
        """
        Create the output directory for storing evaluation results
        if it doesn't exist.
        """
        self.output_dir = os.path.join(os.getcwd(), "filtering_evaluation")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def add_data_point(
        self,
        true_state: np.ndarray,
        measurement: np.ndarray,
        estimated_state: np.ndarray,
        predicted_state: np.ndarray | None = None,
    ) -> None:
        """
        Add a data point to the evaluator.

        Args:
            true_state (np.ndarray): The true state of the system.
            measurement (np.ndarray): The measured state of the system.
            estimated_state (np.ndarray): The estimated state from
            the Kalman filter.
            predicted_state (np.ndarray, optional): The predicted state from
            the Kalman filter.
        """
        self.true_states.append(true_state)
        self.measurements.append(measurement)
        self.estimated_states.append(estimated_state)
        if predicted_state is not None:
            self.predicted_states.append(predicted_state)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the performance of the Kalman filter.

        Returns:
            dict: A dictionary containing various performance metrics.
        """
        true_states = np.array(self.true_states)
        measurements = np.array(self.measurements)
        estimated_states = np.array(self.estimated_states)

        results: Dict[str, Any] = {}

        # Mean Squared Error (MSE)
        mse_measurement = mean_squared_error(true_states, measurements)
        mse_estimate = mean_squared_error(true_states, estimated_states)
        results["MSE_measurement"] = mse_measurement
        results["MSE_estimate"] = mse_estimate
        # MSE improvement percentage
        results["MSE_improvement"] = (
            (mse_measurement - mse_estimate) / mse_measurement * 100
        )

        # Root Mean Squared Error (RMSE)
        results["RMSE_measurement"] = np.sqrt(mse_measurement)
        results["RMSE_estimate"] = np.sqrt(mse_estimate)

        # Mean Absolute Error (MAE)
        results["MAE_measurement"] = np.mean(
            np.abs(true_states - measurements)
        )
        results["MAE_estimate"] = np.mean(
            np.abs(true_states - estimated_states)
        )

        # Innovation (measurement residual) statistics
        innovations = measurements - estimated_states
        results["innovation_mean"] = np.mean(innovations, axis=0)
        results["innovation_std"] = np.std(innovations, axis=0)

        # Consistency check (if predicted states are available)
        if self.predicted_states:
            predicted_states = np.array(self.predicted_states)
            consistency = np.mean(
                (true_states - predicted_states) ** 2, axis=1
            )
            results["prediction_consistency"] = np.mean(consistency)

        return results

    def plot_results(self, state_name: str) -> None:
        """
        Plot the results of the Kalman filter estimation.

        Args:
            state_name (str): The name of the state being estimated.
        """
        true_states = np.array(self.true_states)
        measurements = np.array(self.measurements)
        estimated_states = np.array(self.estimated_states)

        plt.figure(figsize=(12, 8))
        plt.scatter(
            measurements[:, 0],
            measurements[:, 1],
            label="Measurements",
            color="red",
            alpha=0.5,
            s=10,
        )
        plt.plot(
            true_states[:, 0],
            true_states[:, 1],
            label="True State",
            color="green",
        )
        plt.plot(
            estimated_states[:, 0],
            estimated_states[:, 1],
            label="Kalman Filter Estimate",
            color="blue",
        )

        if self.predicted_states:
            predicted_states = np.array(self.predicted_states)
            plt.plot(
                predicted_states[:, 0],
                predicted_states[:, 1],
                label="Kalman Filter Prediction",
                color="orange",
                linestyle="--",
            )

        plt.title(f"{self.filter_name}: {state_name} Estimation")
        plt.legend(loc="upper left", bbox_to_anchor=(0, -0.1))
        plt.xlabel("X Position (units)")
        plt.ylabel("Z Position (units)")
        plt.grid(True)

        # Add text box with statistics
        mse_estimate = mean_squared_error(true_states, estimated_states)
        mse_measurement = mean_squared_error(true_states, measurements)
        # Improvement percentage calculation
        improvement_percentage = (
            (mse_measurement - mse_estimate) / mse_measurement
        ) * 100
        stats_text = f"MSE Improvement: {improvement_percentage:.2f}%\n"
        stats_text += f"Estimation RMSE: {np.sqrt(mse_estimate):.2f}\n"
        stats_text += f"Measurement RMSE: {np.sqrt(mse_measurement):.2f}"
        plt.text(
            0.02,
            0.02,
            stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            verticalalignment="bottom",
        )

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filter_name}_{state_name}_Estimation_{timestamp}.png"
        plt.savefig(
            os.path.join(self.output_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def plot_error(self, state_name: str) -> None:
        """
        Plot the error between true states, estimated states, and measurements.

        Args:
            state_name (str): The name of the state being estimated.
        """
        true_states = np.array(self.true_states)
        estimated_states = np.array(self.estimated_states)
        measurements = np.array(self.measurements)

        # Calculate errors
        estimation_error = np.linalg.norm(
            true_states - estimated_states, axis=1
        )
        measurement_error = np.linalg.norm(true_states - measurements, axis=1)

        # Calculate average errors and RMSE
        avg_estimation_error = np.mean(estimation_error)
        avg_measurement_error = np.mean(measurement_error)
        rmse_estimation = np.sqrt(
            mean_squared_error(true_states, estimated_states)
        )
        rmse_measurement = np.sqrt(
            mean_squared_error(true_states, measurements)
        )

        plt.figure(figsize=(12, 8))
        plt.plot(
            estimation_error,
            label=(
                f"Estimation Error (Avg: {avg_estimation_error:.2f}, "
                f"RMSE: {rmse_estimation:.2f})"
            ),
            color="blue",
        )
        plt.plot(
            measurement_error,
            label=(
                f"Measurement Error (Avg: {avg_measurement_error:.2f}, "
                f"RMSE: {rmse_measurement:.2f})"
            ),
            color="red",
            alpha=0.5,
        )

        plt.title(f"{self.filter_name}: {state_name} Estimation Error")
        plt.xlabel("Time Step")
        plt.ylabel("Error (units)")
        plt.legend(loc="upper left", bbox_to_anchor=(0, -0.1))
        plt.grid(True)

        # Add text box with additional statistics
        # Error reduction percentage calculation
        improvement_percentage = (
            (avg_measurement_error - avg_estimation_error) / avg_measurement_error
        ) * 100
        stats_text = f"Error Reduction: {improvement_percentage:.2f}%\n"
        stats_text += f"Max Estimation Error: {np.max(estimation_error):.2f}\n"
        stats_text += f"Min Estimation Error: {np.min(estimation_error):.2f}"
        plt.text(
            0.02,
            0.02,
            stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            verticalalignment="bottom",
        )

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filter_name}_{state_name}_Error_{timestamp}.png"
        plt.savefig(
            os.path.join(self.output_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def plot_innovations(self) -> None:
        """
        Plot innovation statistics over time.
        """
        innovations = (
                np.array(self.measurements) - np.array(self.estimated_states)
        )
        time_steps = range(len(innovations))

        plt.figure(figsize=(12, 8))
        plt.plot(time_steps, innovations[:, 0], label="X Innovation")
        plt.plot(time_steps, innovations[:, 1], label="Z Innovation")

        plt.axhline(
            y=np.mean(innovations[:, 0]),
            color="r",
            linestyle="--",
            label="X Mean",
        )
        plt.axhline(
            y=np.mean(innovations[:, 1]),
            color="g",
            linestyle="--",
            label="Z Mean",
        )

        plt.title(f"{self.filter_name}: Innovation Statistics")
        plt.xlabel("Time Step")
        plt.ylabel("Innovation")
        plt.legend()
        plt.grid(True)

        # Add text box with statistics
        stats_text = (
            f"X Mean: {np.mean(innovations[:, 0]):.2f}\n"
            f"X Std: {np.std(innovations[:, 0]):.2f}\n"
            f"Z Mean: {np.mean(innovations[:, 1]):.2f}\n"
            f"Z Std: {np.std(innovations[:, 1]):.2f}"
        )
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filter_name}_Innovations_{timestamp}.png"
        plt.savefig(
            os.path.join(self.output_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def plot_consistency(self) -> None:
        """
        Plot prediction consistency over time.
        """
        if not self.predicted_states:
            print("No predicted states available for consistency plot.")
            return

        true_states = np.array(self.true_states)
        predicted_states = np.array(self.predicted_states)
        consistency = np.mean((true_states - predicted_states) ** 2, axis=1)
        time_steps = range(len(consistency))

        plt.figure(figsize=(12, 8))
        plt.plot(time_steps, consistency, label="Prediction Consistency")
        plt.axhline(
            y=np.mean(consistency),
            color="r",
            linestyle="--",
            label="Mean Consistency",
        )

        plt.title(f"{self.filter_name}: Prediction Consistency")
        plt.xlabel("Time Step")
        plt.ylabel("Consistency (MSE)")
        plt.legend()
        plt.grid(True)

        # Add text box with statistics
        stats_text = (
            f"Mean Consistency: {np.mean(consistency):.2f}\n"
            f"Max Consistency: {np.max(consistency):.2f}\n"
            f"Min Consistency: {np.min(consistency):.2f}"
        )
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filter_name}_Consistency_{timestamp}.png"
        plt.savefig(
            os.path.join(self.output_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Get the collected data as a dictionary.

        Returns:
            dict: A dictionary containing the true states, measurements,
                  estimated states, and predicted states (if available).
        """
        return {
            "true_states": np.array(self.true_states),
            "measurements": np.array(self.measurements),
            "estimated_states": np.array(self.estimated_states),
            "predicted_states": np.array(self.predicted_states)
            if self.predicted_states
            else None,
        }

    def run_evaluation(self, state_name: str) -> None:
        """
        Run the full evaluation process, including all plots.

        Args:
            state_name (str): The name of the state being estimated.
        """
        self.evaluate()
        self.plot_results(state_name)
        self.plot_error(state_name)
        self.plot_innovations()
        self.plot_consistency()
