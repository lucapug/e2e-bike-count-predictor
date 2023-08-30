import datetime

import pandas as pd
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnSummaryMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import (
    CounterAgg,
    DashboardPanelCounter,
    DashboardPanelPlot,
    PanelValue,
    PlotType,
    ReportFilter,
)
from evidently.ui.workspace import Workspace, WorkspaceBase

seoul_bike_ref = pd.read_csv('../data/processed/ref_data.csv')
seoul_bike_curr = pd.read_csv('../data/processed/curr_data.csv')


WORKSPACE = "workspace"

PROJECT_NAME = "monitor_bike_count"
PROJECT_DESCRIPTION = "Project using Seoul Bike dataset."


def create_report(i: int):
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="temperature", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="temperature"),
            ColumnDriftMetric(column_name="humidity", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="humidity"),
            ColumnDriftMetric(column_name="rainfall", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="rainfall"),
        ],
        #        timestamp=datetime.datetime.now() + datetime.timedelta(weeks=i),
        timestamp=datetime.datetime(2018, 9, 1) + datetime.timedelta(weeks=i),
    )

    data_drift_report.run(
        reference_data=seoul_bike_ref,
        current_data=seoul_bike_curr.iloc[168 * i : 168 * (i + 1), :],  # 24*7=168
    )
    return data_drift_report


def create_test_suite(i: int):
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        #        timestamp=datetime.datetime.now() + datetime.timedelta(weeks=i),
        timestamp=datetime.datetime(2018, 9, 1) + datetime.timedelta(weeks=i),
    )

    data_drift_test_suite.run(
        reference_data=seoul_bike_ref,
        current_data=seoul_bike_curr.iloc[168 * i : 168 * (i + 1), :],
    )
    return data_drift_test_suite


def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(PROJECT_NAME)
    project.description = PROJECT_DESCRIPTION
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Seoul Rented Bikes dataset",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of Drifted Features",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="share",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset Quality",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="DatasetDriftMetric",
                    field_path="share_of_drifted_columns",
                    legend="Drift Share",
                ),
                PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path=DatasetMissingValuesMetric.fields.current.share_of_missing_values,
                    legend="Missing Values Share",
                ),
            ],
            plot_type=PlotType.LINE,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Temperature: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "temperature"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Humidity: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "humidity"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Rainfall: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "rainfall"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.save()
    return project


def create_bike_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

    for i in range(0, 12):
        report = create_report(i=i)
        ws.add_report(project.id, report)

        test_suite = create_test_suite(i=i)
        ws.add_test_suite(project.id, test_suite)


if __name__ == "__main__":
    create_bike_project(WORKSPACE)
