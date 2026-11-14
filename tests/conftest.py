import pytest

from helpers import make_dataset


@pytest.fixture
def dataset_dir(tmp_path):
    return make_dataset(tmp_path / "dataset")
