import pandas as pd

from release_sales import release


def archive(tmp_path):
    path = tmp_path / "world.csv"
    pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "date": ["20141013T000000", "20150115T000000", "20150220T000000", "20150301T000000", "20150330T000000"],
        "price": [1, 2, 3, 4, 5],
    }).to_csv(path, index=False)
    return path


def test_init_keeps_sales_up_to_the_month(tmp_path):
    live = tmp_path / "live" / "sales.csv"
    assert release(archive(tmp_path), live, init="2015-01") == "2015-01"
    assert list(pd.read_csv(live, dtype=str)["id"]) == ["1", "2"]


def test_each_release_adds_the_next_month_until_none_left(tmp_path):
    world, live = archive(tmp_path), tmp_path / "sales.csv"
    release(world, live, init="2015-01")
    assert release(world, live) == "2015-02"
    assert release(world, live) == "2015-03"
    assert len(pd.read_csv(live)) == 5
    assert release(world, live) is None


def test_same_cutoff_gives_the_same_file(tmp_path):
    world = archive(tmp_path)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    release(world, a, init="2015-02")
    release(world, b, init="2015-01")
    release(world, b)
    assert a.read_bytes() == b.read_bytes()


def test_test_split_is_about_a_fifth():
    from train_model import data_until, is_test_house

    ids = [str(i) for i in range(10000)]
    in_test = [is_test_house(i) for i in ids]
    assert 0.18 < sum(in_test) / len(ids) < 0.22
    assert data_until(1) == "2014-01" and data_until(17) == "2015-05"
