import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Optional


class FiNERProcessor:

    def __init__(self, cache_dir: Optional[str] = None):