"""AnalysisMixin -- statistical analysis methods for DataProcessorMainWindow.

Runs PCA, ANOVA, regression, surface fitting, neural network analysis,
and visualization methods like heatmap and filter comparison.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from .analysis_widgets import FilterComparisonDialog, HeatmapDialog

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """Mixin providing statistical analysis methods."""

    def _show_heatmap(self) -> None:
        """Show heatmap dialog."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        dialog = HeatmapDialog(self.current_data, self)
        dialog.exec()

    def _show_filter_comparison(self) -> None:
        """Show filter comparison dialog."""
        if self.current_data is None or self.filtered_data is None:
            QMessageBox.warning(
                self,
                "No Filtered Data",
                "Apply a filter first to compare original vs filtered.",
            )
            return
        if self.time_column is None:
            QMessageBox.warning(self, "No Time Column", "Set a time column first.")
            return
        signals = self.signal_list.get_selected_signals()
        if not signals:
            QMessageBox.warning(self, "No Signals", "Select signals to compare.")
            return
        dialog = FilterComparisonDialog(
            self.current_data,
            self.filtered_data,
            self.time_column,
            signals,
            self,
        )
        dialog.exec()

    # ── Analysis panel handlers ─────────────────────────────────────────────

    def _run_pca_analysis(self, config: dict) -> None:
        """Run PCA analysis from Analysis tab."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        variables = config.get("variables", [])
        if len(variables) < 2:
            QMessageBox.warning(
                self, "Insufficient Variables", "Select at least 2 variables for PCA."
            )
            return

        try:
            from data_processor.core.pca_analysis import PCAAnalyzer, PCAConfig

            pca_config = PCAConfig(
                n_components=config.get("n_components"),
                standardize=config.get("standardize", True),
                variance_threshold=config.get("variance_threshold", 0.95),
            )
            analyzer = PCAAnalyzer(pca_config)
            result = analyzer.analyze(self.current_data, columns=variables)
            self.analysis_panel.pca_widget.display_results(result)
            self.status_bar.set_status(
                f"PCA complete: {result.n_components} components, "
                f"{result.total_variance_explained:.1%} variance explained"
            )
        except ImportError as e:
            logger.error(f"PCA analysis failed: {e}", exc_info=True)
            QMessageBox.critical(self, "PCA Error", f"Analysis failed:\n{e}")

    def _run_anova_analysis(self, config: dict) -> None:
        """Run ANOVA analysis from Analysis tab."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        try:
            from data_processor.core.anova import (
                ANOVAAnalyzer,
                PostHocMethod,
                format_anova_report,
            )

            alpha = config.get("alpha", 0.05)
            analyzer = ANOVAAnalyzer(alpha=alpha)

            posthoc_map = {
                "Tukey HSD": PostHocMethod.TUKEY_HSD,
                "Bonferroni": PostHocMethod.BONFERRONI,
                "Scheffé": PostHocMethod.SCHEFFE,
                "None": None,
            }
            posthoc = posthoc_map.get(config.get("posthoc", "Tukey HSD"))

            anova_type = config.get("type", "One-Way ANOVA")

            if anova_type == "One-Way ANOVA":
                dependent = config.get("dependent", "")
                group = config.get("group", "")
                if not dependent or not group:
                    QMessageBox.warning(
                        self,
                        "Missing Config",
                        "Select dependent and grouping variables.",
                    )
                    return
                result = analyzer.one_way_anova(
                    self.current_data, dependent, group, post_hoc=posthoc
                )
            elif anova_type == "Two-Way ANOVA":
                dependent = config.get("dependent", "")
                factor_a = config.get("factor_a", "")
                factor_b = config.get("factor_b", "")
                if not dependent or not factor_a or not factor_b:
                    QMessageBox.warning(self, "Missing Config", "Select all variables.")
                    return
                result = analyzer.two_way_anova(
                    self.current_data,
                    dependent,
                    factor_a,
                    factor_b,
                    test_interaction=config.get("interaction", True),
                )
            elif anova_type == "Repeated Measures":
                subject = config.get("subject", "")
                measures = config.get("measures", [])
                if not subject or len(measures) < 2:
                    QMessageBox.warning(
                        self,
                        "Missing Config",
                        "Select subject ID and at least 2 measures.",
                    )
                    return
                result = analyzer.repeated_measures_anova(
                    self.current_data, measures, subject
                )
            else:
                QMessageBox.warning(
                    self,
                    "Unknown Type",
                    f"Unknown ANOVA type: {anova_type}",
                )
                return

            report = format_anova_report(result)
            self.analysis_panel.anova_widget.display_results(report)
            self.status_bar.set_status(f"ANOVA complete ({anova_type})")

        except ImportError as e:
            logger.error(f"ANOVA analysis failed: {e}", exc_info=True)
            QMessageBox.critical(self, "ANOVA Error", f"Analysis failed:\n{e}")

    def _run_regression_analysis(self, config: dict) -> None:
        """Run regression analysis from Analysis tab."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        target = config.get("target", "")
        predictors = config.get("predictors", [])
        if not target or not predictors:
            QMessageBox.warning(
                self, "Missing Config", "Select target and predictor variables."
            )
            return

        try:
            from data_processor.core.regression import (
                MultivariateRegressor,
                RegressionConfig,
                RegularizationType,
                SelectionMethod,
                format_regression_report,
            )

            reg_map = {
                "none": RegularizationType.NONE,
                "ridge": RegularizationType.RIDGE,
                "lasso": RegularizationType.LASSO,
                "elastic net": RegularizationType.ELASTIC_NET,
            }
            sel_map = {
                "none": SelectionMethod.NONE,
                "forward": SelectionMethod.FORWARD,
                "backward": SelectionMethod.BACKWARD,
                "stepwise": SelectionMethod.STEPWISE,
            }

            reg_config = RegressionConfig(
                regularization=reg_map.get(
                    config.get("regularization", "none"), RegularizationType.NONE
                ),
                alpha=config.get("alpha", 1.0),
                polynomial_degree=config.get("polynomial_degree", 1),
                include_interactions=config.get("interactions", False),
                selection_method=sel_map.get(
                    config.get("selection", "none"), SelectionMethod.NONE
                ),
            )
            regressor = MultivariateRegressor(reg_config)
            result = regressor.fit(self.current_data, target, predictors)
            report = format_regression_report(result)
            self.analysis_panel.regression_widget.display_results(result, report)
            self.status_bar.set_status(
                f"Regression complete: R² = {result.r_squared:.4f}"
            )

        except ImportError as e:
            logger.error(f"Regression analysis failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Regression Error", f"Analysis failed:\n{e}")

    def _run_surface_analysis(self, config: dict) -> None:
        """Run surface plot from Analysis tab."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        x_col = config.get("x_column", "")
        y_col = config.get("y_column", "")
        z_col = config.get("z_column", "")
        if not x_col or not y_col or not z_col:
            QMessageBox.warning(self, "Missing Config", "Select X, Y, and Z columns.")
            return

        try:
            import matplotlib.pyplot as plt

            from data_processor.core.surface_plot import (
                InterpolationMethod,
                SmoothingMethod,
                SurfacePlotConfig,
                SurfacePlotEngine,
            )

            interp_str = config.get("interpolation", "linear")
            smooth_str = config.get("smoothing", "none")

            plot_config = SurfacePlotConfig(
                x_column=x_col,
                y_column=y_col,
                z_column=z_col,
                grid_resolution=config.get("grid_resolution", 50),
                interpolation_method=InterpolationMethod(interp_str),
                smoothing_method=SmoothingMethod(smooth_str),
                smoothing_sigma=config.get("smoothing_sigma", 1.0),
                smoothing_kernel_size=config.get("smoothing_kernel", 3),
                remove_outliers=config.get("remove_outliers", False),
                outlier_threshold=config.get("outlier_threshold", 3.0),
                colormap=config.get("colormap", "viridis"),
                alpha=config.get("alpha", 0.8),
                show_scatter=config.get("show_scatter", True),
                title=f"Surface: {z_col} vs ({x_col}, {y_col})",
                x_label=x_col,
                y_label=y_col,
                z_label=z_col,
            )

            engine = SurfacePlotEngine()
            result = engine.create_surface(self.current_data, plot_config)

            # Create matplotlib figure for the surface
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(
                result.x_grid,
                result.y_grid,
                result.z_grid,
                cmap=plot_config.colormap,
                alpha=plot_config.alpha,
            )
            if plot_config.show_scatter:
                ax.scatter(
                    result.x_data,
                    result.y_data,
                    result.z_data,
                    c="red",
                    s=5,
                    alpha=0.3,
                )
            ax.set_xlabel(plot_config.x_label)
            ax.set_ylabel(plot_config.y_label)
            ax.set_zlabel(plot_config.z_label)
            ax.set_title(plot_config.title)
            plt.show()
            self.status_bar.set_status("Surface plot generated")

        except ImportError as e:
            logger.error(f"Surface plot failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Surface Plot Error", f"Plot failed:\n{e}")

    def _run_nn_analysis(self, config: dict) -> None:
        """Run neural network training from Analysis tab."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        try:
            from data_processor.core.neural_network import NeuralNetworkTrainer

            trainer = NeuralNetworkTrainer(config)
            result = trainer.train(self.current_data)
            self.analysis_panel.nn_widget.display_results(result)
            self.status_bar.set_status("Neural network training complete")

        except ImportError as e:
            logger.error(f"Neural network training failed: {e}", exc_info=True)
            QMessageBox.critical(self, "NN Error", f"Training failed:\n{e}")
