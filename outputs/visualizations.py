"""
Visualization module using Plotly for interactive charts.
Generates charts for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates interactive Plotly charts for bond fund analytics.
    """

    # Color palette
    COLORS = {
        'primary': '#1f4e79',
        'secondary': '#2e75b6',
        'positive': '#92d050',
        'negative': '#ff6b6b',
        'neutral': '#ffc000',
        'categories': px.colors.qualitative.Set2,
    }

    def __init__(self):
        """Initialize the chart generator."""
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif'},
            'hovermode': 'x unified',
        }

    def cumulative_returns_chart(
        self,
        returns_df: pd.DataFrame,
        title: str = "Cumulative Returns (Growth of $1)"
    ) -> go.Figure:
        """
        Create cumulative returns line chart.

        Args:
            returns_df: DataFrame with monthly returns
            title: Chart title

        Returns:
            Plotly Figure
        """
        cumulative = (1 + returns_df).cumprod()

        fig = go.Figure()

        for i, col in enumerate(cumulative.columns):
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative[col],
                name=col,
                mode='lines',
                line=dict(width=2),
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Growth of $1',
            legend_title='Fund',
            **self.default_layout
        )

        return fig

    def yield_vs_duration_scatter(
        self,
        fund_facts_df: pd.DataFrame,
        title: str = "Yield vs Duration"
    ) -> go.Figure:
        """
        Create yield vs duration scatter plot.
        Size represents AUM, color represents category.

        Args:
            fund_facts_df: DataFrame with fund characteristics
            title: Chart title

        Returns:
            Plotly Figure
        """
        df = fund_facts_df.reset_index()

        # Check required columns exist and have data
        if 'duration' not in df.columns or 'yield' not in df.columns:
            return self._empty_chart("Duration and yield data required for this chart")

        # Filter out rows with missing duration or yield
        df = df.dropna(subset=['duration', 'yield'])
        if df.empty:
            return self._empty_chart("No funds with both duration and yield data")

        # Handle missing AUM values - use default size if all NaN
        if 'aum' in df.columns and df['aum'].notna().any():
            median_aum = df['aum'].median()
            df['size'] = df['aum'].fillna(median_aum) / 1e9  # In billions
            df['size'] = df['size'].clip(lower=0.1)  # Ensure minimum size
        else:
            df['size'] = 10  # Default size

        fig = px.scatter(
            df,
            x='duration',
            y='yield',
            size='size',
            color='category' if 'category' in df.columns else None,
            hover_name='ticker' if 'ticker' in df.columns else None,
            hover_data=['name'] if 'name' in df.columns else None,
            title=title,
            labels={
                'duration': 'Effective Duration (years)',
                'yield': 'Yield',
                'size': 'AUM ($B)',
            },
            color_discrete_sequence=self.COLORS['categories'],
        )

        fig.update_layout(**self.default_layout)
        fig.update_traces(marker=dict(sizemin=5))

        return fig

    def sharpe_vs_drawdown_scatter(
        self,
        metrics_df: pd.DataFrame,
        title: str = "Sharpe Ratio vs Maximum Drawdown"
    ) -> go.Figure:
        """
        Create Sharpe ratio vs max drawdown scatter plot.
        Efficient funds are in upper-left (high Sharpe, low drawdown).

        Args:
            metrics_df: DataFrame with returns metrics
            title: Chart title

        Returns:
            Plotly Figure
        """
        df = metrics_df.reset_index()

        fig = px.scatter(
            df,
            x='max_drawdown',
            y='sharpe_ratio',
            hover_name='ticker' if 'ticker' in df.columns else df.index,
            text='ticker' if 'ticker' in df.columns else None,
            title=title,
            labels={
                'max_drawdown': 'Maximum Drawdown',
                'sharpe_ratio': 'Sharpe Ratio',
            },
        )

        # Add quadrant lines
        if not df.empty:
            median_sharpe = df['sharpe_ratio'].median()
            median_dd = df['max_drawdown'].median()

            fig.add_hline(y=median_sharpe, line_dash='dash', line_color='gray', opacity=0.5)
            fig.add_vline(x=median_dd, line_dash='dash', line_color='gray', opacity=0.5)

            # Add annotations for quadrants
            fig.add_annotation(
                x=df['max_drawdown'].min() * 0.5,
                y=df['sharpe_ratio'].max() * 0.95,
                text="Best<br>(High Sharpe, Low DD)",
                showarrow=False,
                font=dict(color='green', size=10),
            )

        fig.update_traces(
            marker=dict(size=12, color=self.COLORS['primary']),
            textposition='top center'
        )
        fig.update_layout(**self.default_layout)

        return fig

    def downside_capture_scatter(
        self,
        metrics_df: pd.DataFrame,
        title: str = "Downside Capture vs Volatility"
    ) -> go.Figure:
        """
        Create downside capture vs volatility scatter plot.
        Best funds are in lower-left quadrant.

        Args:
            metrics_df: DataFrame with returns metrics
            title: Chart title

        Returns:
            Plotly Figure
        """
        df = metrics_df.reset_index()

        if 'downside_capture' not in df.columns or 'volatility' not in df.columns:
            return self._empty_chart("Missing required columns for this chart")

        fig = px.scatter(
            df,
            x='volatility',
            y='downside_capture',
            hover_name='ticker' if 'ticker' in df.columns else df.index,
            text='ticker' if 'ticker' in df.columns else None,
            title=title,
            labels={
                'volatility': 'Volatility (Annualized)',
                'downside_capture': 'Downside Capture (%)',
            },
        )

        fig.update_traces(
            marker=dict(size=12, color=self.COLORS['secondary']),
            textposition='top center'
        )
        fig.update_layout(**self.default_layout)

        return fig

    def credit_quality_bar(
        self,
        fund_facts_df: pd.DataFrame,
        ticker: Optional[str] = None,
        title: str = "Credit Quality Breakdown"
    ) -> go.Figure:
        """
        Create stacked bar chart of credit quality breakdown.

        Args:
            fund_facts_df: DataFrame with fund facts including quality_breakdown
            ticker: Optional specific ticker to show
            title: Chart title

        Returns:
            Plotly Figure
        """
        # This requires quality_breakdown data which might be in a nested dict
        # For now, create a placeholder if data not available
        fig = go.Figure()

        rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Below B', 'Not Rated']
        colors = px.colors.sequential.RdYlGn_r

        # Sample data structure - in practice this would come from fund_facts_df
        fig.update_layout(
            title=title,
            barmode='stack',
            xaxis_title='Fund',
            yaxis_title='Allocation (%)',
            **self.default_layout
        )

        return fig

    def sector_allocation_pie(
        self,
        fund_facts: Dict[str, float],
        ticker: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create pie chart for sector allocation.

        Args:
            fund_facts: Dictionary with sector weights
            ticker: Fund ticker for title
            title: Optional custom title

        Returns:
            Plotly Figure
        """
        if not fund_facts:
            return self._empty_chart("No sector data available")

        labels = list(fund_facts.keys())
        values = list(fund_facts.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
        )])

        fig.update_layout(
            title=title or f"{ticker} Sector Allocation",
            **self.default_layout
        )

        return fig

    def scenario_impact_bar(
        self,
        scenario_df: pd.DataFrame,
        scenario_col: str,
        title: str = "Scenario Impact Analysis"
    ) -> go.Figure:
        """
        Create horizontal bar chart for scenario impacts.

        Args:
            scenario_df: DataFrame with scenario results
            scenario_col: Column name for the scenario to display
            title: Chart title

        Returns:
            Plotly Figure
        """
        df = scenario_df.reset_index()

        if scenario_col not in df.columns:
            return self._empty_chart(f"Column {scenario_col} not found")

        df = df.dropna(subset=[scenario_col]).sort_values(scenario_col)

        if df.empty:
            return self._empty_chart("No data available for this scenario")

        colors = [self.COLORS['negative'] if x < 0 else self.COLORS['positive']
                  for x in df[scenario_col]]

        fig = go.Figure(go.Bar(
            x=df[scenario_col],
            y=df['ticker'] if 'ticker' in df.columns else df.index,
            orientation='h',
            marker_color=colors,
            text=[f"{x:.2f}%" for x in df[scenario_col]],
            textposition='outside',
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Price Impact (%)',
            yaxis_title='Fund',
            **self.default_layout
        )

        return fig

    def rolling_metrics_chart(
        self,
        rolling_df: pd.DataFrame,
        metric: str = 'rolling_sharpe',
        title: str = "Rolling Metrics"
    ) -> go.Figure:
        """
        Create line chart for rolling metrics.

        Args:
            rolling_df: DataFrame with rolling metrics
            metric: Metric column to plot
            title: Chart title

        Returns:
            Plotly Figure
        """
        if metric not in rolling_df.columns:
            return self._empty_chart(f"Metric {metric} not found")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rolling_df.index,
            y=rolling_df[metric],
            mode='lines',
            line=dict(color=self.COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 78, 121, 0.1)',
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=metric.replace('_', ' ').title(),
            **self.default_layout
        )

        return fig

    def correlation_heatmap(
        self,
        returns_df: pd.DataFrame,
        title: str = "Return Correlations"
    ) -> go.Figure:
        """
        Create correlation heatmap for fund returns.

        Args:
            returns_df: DataFrame with returns
            title: Chart title

        Returns:
            Plotly Figure
        """
        corr = returns_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
        ))

        fig.update_layout(
            title=title,
            xaxis_title='',
            yaxis_title='',
            **self.default_layout
        )

        return fig

    def drawdown_chart(
        self,
        returns_df: pd.DataFrame,
        fund_col: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create drawdown chart over time.

        Args:
            returns_df: DataFrame with returns
            fund_col: Column name for the fund
            title: Chart title

        Returns:
            Plotly Figure
        """
        if fund_col not in returns_df.columns:
            return self._empty_chart(f"Fund {fund_col} not found")

        returns = returns_df[fund_col].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,  # Convert to percentage
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.3)',
            line=dict(color=self.COLORS['negative'], width=1),
        ))

        fig.update_layout(
            title=title or f"{fund_col} Drawdown",
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            **self.default_layout
        )

        return fig

    def metrics_comparison_radar(
        self,
        metrics_df: pd.DataFrame,
        tickers: List[str],
        metrics: List[str] = ['sharpe_ratio', 'alpha', 'upside_capture', 'information_ratio'],
        title: str = "Fund Comparison"
    ) -> go.Figure:
        """
        Create radar chart comparing multiple funds.

        Args:
            metrics_df: DataFrame with metrics
            tickers: List of tickers to compare
            metrics: List of metrics to include
            title: Chart title

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        for ticker in tickers:
            if ticker not in metrics_df.index:
                continue

            values = []
            for metric in metrics:
                if metric in metrics_df.columns:
                    val = metrics_df.loc[ticker, metric]
                    # Handle NaN values
                    if pd.isna(val):
                        values.append(50)  # Default to middle
                        continue
                    # Normalize to 0-100 scale
                    col_min = metrics_df[metric].min()
                    col_max = metrics_df[metric].max()
                    if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
                        normalized = (val - col_min) / (col_max - col_min) * 100
                    else:
                        normalized = 50
                    values.append(normalized)
                else:
                    values.append(0)

            # Close the radar
            values.append(values[0])
            metrics_closed = metrics + [metrics[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_closed,
                fill='toself',
                name=ticker,
            ))

        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            **self.default_layout
        )

        return fig

    def _empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='gray'),
        )
        fig.update_layout(**self.default_layout)
        return fig
