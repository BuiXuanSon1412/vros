import streamlit as st
import streamlit.components.v1 as components
from utils.visualizer import Visualizer


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_map_view(problem_type):
    """Render map view with routes"""
    customers = st.session_state.get(f"customers_{problem_type}")
    depot = st.session_state.get(f"depot_{problem_type}")
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if customers is None or depot is None:
        st.info("Generate or upload data to see the map")
        return

    viz = get_visualizer()

    # ============= WITH SOLUTION ==============
    if solution is not None and solution.get("routes"):
        fig = viz.plot_routes_2d(
            customers,
            depot,
            solution["routes"],
            title=f"Routes - {solution['algorithm']}",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_{problem_type}_{chart_counter}",
        )

        # Inject hover highlight JavaScript
        components.html(
            """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const plots = document.getElementsByClassName("js-plotly-plot");
    if (plots.length === 0) return;

    const plot = plots[plots.length - 1]; // most recent plot

    plot.on('plotly_hover', function(e) {
        if (plot._js_on_hover) {
            plot._js_on_hover(e);
        }
    });

    plot.on('plotly_unhover', function(e) {
        if (plot._js_on_unhover) {
            plot._js_on_unhover(e);
        }
    });
});
</script>
            """,
            height=0,
        )

    # ============= WITHOUT SOLUTION ==============
    else:
        fig = viz.plot_routes_2d(
            customers,
            depot,
            {},
            title="Customer Locations",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_empty_{problem_type}",
        )

        # Still attach events so future features won't break
        components.html(
            """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const plots = document.getElementsByClassName("js-plotly-plot");
    if (plots.length === 0) return;

    const plot = plots[plots.length - 1];

    plot.on('plotly_hover', function(e) {
        if (plot._js_on_hover) {
            plot._js_on_hover(e);
        }
    });

    plot.on('plotly_unhover', function(e) {
        if (plot._js_on_unhover) {
            plot._js_on_unhover(e);
        }
    });
});
</script>
            """,
            height=0,
        )
