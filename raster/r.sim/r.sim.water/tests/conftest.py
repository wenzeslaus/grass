import pytest

@pytest.fixture
def session(tmp_path):
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        yield session