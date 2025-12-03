import pandas as pd
import numpy as np
import pyarrow as pa
from utils import e_dist

df = pd.read_parquet('test.parquet')
