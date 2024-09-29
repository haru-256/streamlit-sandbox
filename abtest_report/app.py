import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st

from data.data import BayesianSampler, BinaryBayesianSampler, HyperParams

n_samples = 10000


def sampler_factory(seed: int, dist_type: str, prior_hyper_param: HyperParams) -> BayesianSampler:
    if dist_type == "binary":
        sampler = BinaryBayesianSampler(seed=seed, **prior_hyper_param)
    else:
        raise NotImplementedError(f"Invalid dist_type: {dist_type}")
    return sampler


@st.cache_data
def sampling(
    seed: int,
    control_params: HyperParams,
    treatment_params: HyperParams,
    n_samples: int,
    dist_type: str = "binary",
) -> pl.DataFrame:
    control = sampler_factory(seed, dist_type, control_params)
    treatment = sampler_factory(seed, dist_type, treatment_params)
    return pl.DataFrame(
        {
            "group": ["A"] * n_samples + ["B"] * n_samples,
            "value": np.concatenate(
                [control.sampling_mean(n_samples), treatment.sampling_mean(n_samples)], axis=0
            ),
        }
    )


# Generate random data
df = sampling(
    seed=1026,
    control_params={"alpha": 100, "beta": 50},
    treatment_params={"alpha": 120, "beta": 30},
    n_samples=n_samples,
    dist_type="binary",
).to_pandas()
# df.pivot
# Create a histogram
fig = px.histogram(
    df,
    x="value",
    marginal="box",
    title="A/B mean distribution",
    color="group",
    nbins=100,
    barmode="overlay",
)
st.plotly_chart(fig)
