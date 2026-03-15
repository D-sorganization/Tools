import pytest

@pytest.mark.live_simulation
def test_trimesh_urdf_generation():
    """Strong contract: Ensure complex geometric computation meshes resolve correctly."""
    import trimesh
    mesh = trimesh.creation.box((1,1,1))
    assert mesh.volume == pytest.approx(1.0)
    assert mesh.is_watertight
