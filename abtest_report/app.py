import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st

from data.data import Sampler

n_samples = 10000


def sampler_factory(seed: int, mean: float, std: float) -> Sampler:
    return Sampler(seed=seed, mean=mean, std=std)


@st.cache_data
def sampling(
    seed: int,
    control_params: tuple[float, float],
    treatment_params: tuple[float, float],
    n_samples: int,
) -> pl.DataFrame:
    control = sampler_factory(seed, mean=control_params[0], std=control_params[1])
    treatment = sampler_factory(seed, mean=treatment_params[0], std=treatment_params[1])
    return pl.DataFrame(
        {
            "group": ["A"] * n_samples + ["B"] * n_samples,
            "value": np.concatenate(
                [control.gen_data(n_samples), treatment.gen_data(n_samples)], axis=0
            ),
        }
    )


# Generate random data
df = sampling(1026, (0, 1), (0.3, 1), n_samples)
# df.pivot
# Create a histogram
fig = px.histogram(
    df, x="value", marginal="box", title="A/B mean distribution", color="group", nbins=100
)
st.plotly_chart(fig)
