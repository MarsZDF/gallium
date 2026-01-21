"""Tests for experiment tracking functionality."""

import json

import pytest

import gallium
from gallium import Experiment, Tracker
from gallium.exceptions import InvalidFilterError, SerializationError


class TestTracker:
    """Tests for the Tracker class."""

    def test_tracker_initialization(self, tmp_path):
        """Test tracker initializes lazily."""
        db_path = tmp_path / "test.db"
        tracker = Tracker(db_path)

        # DB should not exist yet (lazy init)
        assert not db_path.exists()

        # After logging, DB should exist
        tracker.log(prompt="test")
        assert db_path.exists()

    def test_log_basic(self, tmp_path):
        """Test basic experiment logging."""
        tracker = Tracker(tmp_path / "test.db")

        exp_id = tracker.log(
            prompt="cyberpunk city",
            seed=42,
            model="flux.2-pro",
        )

        assert exp_id == 1

        experiments = tracker.find()
        assert len(experiments) == 1
        assert experiments[0].prompt == "cyberpunk city"
        assert experiments[0].seed == 42
        assert experiments[0].model == "flux.2-pro"

    def test_log_with_params(self, tmp_path):
        """Test logging with additional params."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(
            prompt="test",
            params={"guidance": 7.5, "steps": 50}
        )

        exp = tracker.find()[0]
        assert exp.params == {"guidance": 7.5, "steps": 50}

    def test_log_invalid_params_raises(self, tmp_path):
        """Test that non-serializable params raise SerializationError."""
        tracker = Tracker(tmp_path / "test.db")

        with pytest.raises(SerializationError):
            tracker.log(prompt="test", params={"func": lambda x: x})

    def test_log_many(self, tmp_path):
        """Test batch logging."""
        tracker = Tracker(tmp_path / "test.db")

        ids = tracker.log_many([
            {"prompt": "cat", "seed": 1},
            {"prompt": "cat", "seed": 2},
            {"prompt": "dog", "seed": 3},
        ])

        assert len(ids) == 3
        assert ids == [1, 2, 3]

        experiments = tracker.find(prompt="cat")
        assert len(experiments) == 2

    def test_find_by_prompt_contains(self, tmp_path):
        """Test finding by prompt substring."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="cyberpunk city at night")
        tracker.log(prompt="forest in morning")
        tracker.log(prompt="cyberpunk alley")

        results = tracker.find(prompt__contains="cyberpunk")
        assert len(results) == 2

    def test_find_by_seed(self, tmp_path):
        """Test finding by exact seed."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test1", seed=42)
        tracker.log(prompt="test2", seed=123)
        tracker.log(prompt="test3", seed=42)

        results = tracker.find(seed=42)
        assert len(results) == 2

    def test_find_by_model(self, tmp_path):
        """Test finding by model name."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test1", model="flux.2-pro")
        tracker.log(prompt="test2", model="sdxl")
        tracker.log(prompt="test3", model="flux.2-pro")

        results = tracker.find(model="flux.2-pro")
        assert len(results) == 2

    def test_find_invalid_filter_raises(self, tmp_path):
        """Test that invalid filter raises InvalidFilterError."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test")

        with pytest.raises(InvalidFilterError):
            tracker.find(invalid_filter="value")

    def test_find_empty_db_returns_empty_list(self, tmp_path):
        """Test that find on non-existent DB returns empty list."""
        tracker = Tracker(tmp_path / "nonexistent.db")

        # Should not raise, should return empty list
        results = tracker.find(prompt__contains="test")
        assert results == []

    def test_recent(self, tmp_path):
        """Test getting recent experiments."""
        tracker = Tracker(tmp_path / "test.db")

        for i in range(10):
            tracker.log(prompt=f"test{i}")

        recent = tracker.recent(5)
        assert len(recent) == 5

        # Should be in reverse order (most recent first)
        assert recent[0].prompt == "test9"
        assert recent[4].prompt == "test5"

    def test_recent_empty_db_returns_empty_list(self, tmp_path):
        """Test that recent on non-existent DB returns empty list."""
        tracker = Tracker(tmp_path / "nonexistent.db")

        results = tracker.recent(10)
        assert results == []

    def test_export_csv(self, tmp_path):
        """Test CSV export."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test1", seed=1)
        tracker.log(prompt="test2", seed=2)

        export_path = tmp_path / "export.csv"
        result_path = tracker.export("csv", str(export_path))

        assert result_path == str(export_path)
        assert export_path.exists()

        # Verify content
        content = export_path.read_text()
        assert "test1" in content
        assert "test2" in content

    def test_export_json(self, tmp_path):
        """Test JSON export."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test1", seed=1, params={"key": "value"})

        export_path = tmp_path / "export.json"
        tracker.export("json", str(export_path))

        data = json.loads(export_path.read_text())
        assert len(data) == 1
        assert data[0]["prompt"] == "test1"
        assert data[0]["params"] == {"key": "value"}

    def test_export_invalid_format_raises(self, tmp_path):
        """Test that invalid export format raises ValueError."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test")

        with pytest.raises(ValueError):
            tracker.export("xml")

    def test_find_with_pagination(self, tmp_path):
        """Test find with limit and offset."""
        tracker = Tracker(tmp_path / "test.db")

        for i in range(20):
            tracker.log(prompt=f"test{i}")

        # Get first page
        page1 = tracker.find(limit=5, offset=0)
        assert len(page1) == 5
        assert page1[0].prompt == "test19"  # Most recent first

        # Get second page
        page2 = tracker.find(limit=5, offset=5)
        assert len(page2) == 5
        assert page2[0].prompt == "test14"

        # Get all
        all_results = tracker.find()
        assert len(all_results) == 20

    def test_context_manager(self, tmp_path):
        """Test tracker as context manager."""
        db_path = tmp_path / "test.db"

        with Tracker(db_path) as tracker:
            tracker.log(prompt="test")
            assert tracker._local.conn is not None

        # Connection should be closed after context
        assert not hasattr(tracker._local, "conn") or tracker._local.conn is None

    def test_context_manager_on_exception(self, tmp_path):
        """Test context manager closes connection on exception."""
        db_path = tmp_path / "test.db"

        try:
            with Tracker(db_path) as tracker:
                tracker.log(prompt="test")
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        # Connection should be closed even after exception
        assert not hasattr(tracker._local, "conn") or tracker._local.conn is None


class TestGlobalAPI:
    """Tests for the global API functions."""

    def test_init_custom_path(self, tmp_path, monkeypatch):
        """Test init with custom database path."""
        # Change to temp directory to avoid polluting the real cwd
        monkeypatch.chdir(tmp_path)

        db_path = tmp_path / "custom.db"
        gallium.init(str(db_path))

        gallium.log(prompt="test")

        assert db_path.exists()

    def test_log_and_find(self, tmp_path, monkeypatch):
        """Test global log and find functions."""
        monkeypatch.chdir(tmp_path)

        # Reset global tracker
        gallium._track._default_tracker = None

        gallium.log(prompt="test prompt", seed=42)

        results = gallium.find(seed=42)
        assert len(results) == 1
        assert results[0].prompt == "test prompt"


class TestExperiment:
    """Tests for the Experiment dataclass."""

    def test_experiment_creation(self):
        """Test creating an Experiment."""
        exp = Experiment(
            id=1,
            prompt="test",
            seed=42,
            model="flux.2-pro",
        )

        assert exp.id == 1
        assert exp.prompt == "test"
        assert exp.seed == 42
        assert exp.params == {}  # Default empty dict

    def test_experiment_load_image_no_path_raises(self):
        """Test that load_image raises when path is None."""
        exp = Experiment(id=1, prompt="test")

        with pytest.raises(FileNotFoundError):
            exp.load_image()

    def test_experiment_repr(self):
        """Test Experiment __repr__ output."""
        exp = Experiment(
            id=1,
            prompt="cyberpunk city",
            seed=42,
            model="flux.2-pro",
        )

        repr_str = repr(exp)
        assert "Experiment(" in repr_str
        assert "id=1" in repr_str
        assert "prompt='cyberpunk city'" in repr_str
        assert "seed=42" in repr_str
        assert "model='flux.2-pro'" in repr_str

    def test_experiment_repr_minimal(self):
        """Test Experiment __repr__ with minimal fields."""
        exp = Experiment(id=1, prompt="test")

        repr_str = repr(exp)
        assert "Experiment(" in repr_str
        assert "id=1" in repr_str
        assert "prompt='test'" in repr_str
        # Optional fields should not appear
        assert "seed=" not in repr_str
        assert "model=" not in repr_str


class TestStarredAndNotes:
    """Tests for starred and notes functionality."""

    def test_log_with_starred(self, tmp_path):
        """Test logging experiments with starred flag."""
        tracker = Tracker(tmp_path / "test.db")

        exp_id = tracker.log(prompt="great result", seed=42, starred=True)
        exp = tracker.find(starred=True)[0]

        assert exp.id == exp_id
        assert exp.starred is True

    def test_log_with_notes(self, tmp_path):
        """Test logging experiments with notes."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test", notes="Initial experiment")
        exp = tracker.find()[0]

        assert exp.notes == "Initial experiment"

    def test_star_method(self, tmp_path):
        """Test star() method for updating experiments."""
        tracker = Tracker(tmp_path / "test.db")

        exp_id = tracker.log(prompt="test")

        # Initially not starred
        exp = tracker.find()[0]
        assert exp.starred is False

        # Star it
        result = tracker.star(exp_id, True)
        assert result is True

        exp = tracker.find()[0]
        assert exp.starred is True

        # Unstar it
        tracker.star(exp_id, False)
        exp = tracker.find()[0]
        assert exp.starred is False

    def test_annotate_method(self, tmp_path):
        """Test annotate() method for updating notes."""
        tracker = Tracker(tmp_path / "test.db")

        exp_id = tracker.log(prompt="test")

        # Add notes
        result = tracker.annotate(exp_id, "Best result!")
        assert result is True

        exp = tracker.find()[0]
        assert exp.notes == "Best result!"

        # Update notes
        tracker.annotate(exp_id, "Actually second best")
        exp = tracker.find()[0]
        assert exp.notes == "Actually second best"

    def test_find_by_starred(self, tmp_path):
        """Test finding experiments by starred status."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="good", starred=True)
        tracker.log(prompt="ok", starred=False)
        tracker.log(prompt="great", starred=True)

        starred = tracker.find(starred=True)
        unstarred = tracker.find(starred=False)

        assert len(starred) == 2
        assert len(unstarred) == 1
        assert all(e.starred for e in starred)

    def test_find_by_notes_contains(self, tmp_path):
        """Test finding experiments by notes content."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test1", notes="Great lighting")
        tracker.log(prompt="test2", notes="Bad composition")
        tracker.log(prompt="test3", notes="Great colors")

        results = tracker.find(notes__contains="Great")
        assert len(results) == 2


class TestDelete:
    """Tests for delete functionality."""

    def test_delete_experiment(self, tmp_path):
        """Test deleting an experiment."""
        tracker = Tracker(tmp_path / "test.db")

        exp_id = tracker.log(prompt="to delete")
        assert len(tracker.find()) == 1

        result = tracker.delete(exp_id)
        assert result is True
        assert len(tracker.find()) == 0

    def test_delete_nonexistent(self, tmp_path):
        """Test deleting non-existent experiment returns False."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test")

        result = tracker.delete(9999)
        assert result is False

    def test_delete_on_empty_db(self, tmp_path):
        """Test delete on non-existent database returns False."""
        tracker = Tracker(tmp_path / "nonexistent.db")

        result = tracker.delete(1)
        assert result is False


class TestHTMLExport:
    """Tests for HTML export functionality."""

    def test_export_html_basic(self, tmp_path):
        """Test basic HTML export."""
        tracker = Tracker(tmp_path / "test.db")

        tracker.log(prompt="test1", seed=1)
        tracker.log(prompt="test2", seed=2, starred=True, notes="Good one")

        export_path = tmp_path / "gallery.html"
        result = tracker.export("html", str(export_path))

        assert result == str(export_path)
        assert export_path.exists()

        content = export_path.read_text()
        assert "test1" in content
        assert "test2" in content
        assert "★" in content  # Star icon
        assert "Good one" in content  # Notes

    def test_export_html_custom_title(self, tmp_path):
        """Test HTML export with custom title."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test")

        export_path = tmp_path / "gallery.html"
        tracker.export("html", str(export_path), title="My Custom Gallery")

        content = export_path.read_text()
        assert "My Custom Gallery" in content

    def test_export_invalid_format_raises(self, tmp_path):
        """Test that invalid export format raises ValueError."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test")

        with pytest.raises(ValueError, match="Unsupported format"):
            tracker.export("xml")


class TestSweep:
    """Tests for the sweep helper function."""

    def test_sweep_basic(self):
        """Test basic sweep functionality."""
        params = gallium.sweep("seed", [1, 2, 3])

        assert len(params) == 3
        assert params[0] == {"seed": 1}
        assert params[1] == {"seed": 2}
        assert params[2] == {"seed": 3}

    def test_sweep_with_base_params(self):
        """Test sweep with base parameters."""
        base = {"prompt": "test", "model": "flux"}
        params = gallium.sweep("seed", [1, 2], base)

        assert len(params) == 2
        assert params[0] == {"prompt": "test", "model": "flux", "seed": 1}
        assert params[1] == {"prompt": "test", "model": "flux", "seed": 2}

    def test_sweep_with_range(self):
        """Test sweep with range object."""
        params = gallium.sweep("guidance", range(5, 8))

        assert len(params) == 3
        assert [p["guidance"] for p in params] == [5, 6, 7]

    def test_sweep_with_floats(self):
        """Test sweep with float values."""
        params = gallium.sweep("cfg", [5.0, 7.5, 10.0])

        assert params[0]["cfg"] == 5.0
        assert params[1]["cfg"] == 7.5
        assert params[2]["cfg"] == 10.0

    def test_sweep_does_not_mutate_base(self):
        """Test that sweep doesn't mutate the base params dict."""
        base = {"prompt": "test"}
        gallium.sweep("seed", [1, 2], base)

        assert base == {"prompt": "test"}  # Unchanged


class TestMigration:
    """Tests for database schema migration."""

    def test_migration_v1_to_v2(self, tmp_path):
        """Test that v1 database migrates to v2 correctly."""
        import sqlite3

        db_path = tmp_path / "test.db"

        # Create a v1 database (without starred and notes columns)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                seed INTEGER,
                path TEXT,
                model TEXT,
                width INTEGER,
                height INTEGER,
                duration_ms INTEGER,
                image_hash TEXT,
                params TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY)
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute(
            "INSERT INTO experiments (prompt, created_at) VALUES (?, ?)",
            ("old experiment", "2024-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        # Now open with Tracker - should migrate
        tracker = Tracker(db_path)
        experiments = tracker.find()

        # Old experiment should be accessible
        assert len(experiments) == 1
        assert experiments[0].prompt == "old experiment"
        # New columns should have defaults
        assert experiments[0].starred is False
        assert experiments[0].notes is None

        # Should be able to use new features
        tracker.star(experiments[0].id, True)
        tracker.annotate(experiments[0].id, "migrated!")

        experiments = tracker.find()
        assert experiments[0].starred is True
        assert experiments[0].notes == "migrated!"

        # Verify schema version was updated
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        version = cursor.fetchone()[0]
        conn.close()
        assert version == 2


class TestGlobalLogExtended:
    """Tests for global log() API with starred/notes."""

    def test_log_with_starred_and_notes(self, tmp_path):
        """Test global log() accepts starred and notes params."""
        gallium.init(tmp_path / "test.db")

        gallium.log(
            prompt="test",
            starred=True,
            notes="important experiment",
        )

        experiments = gallium.find()
        assert len(experiments) == 1
        assert experiments[0].starred is True
        assert experiments[0].notes == "important experiment"


class TestExperimentLoadImage:
    """Tests for Experiment.load_image() path validation."""

    def test_load_image_nonexistent_path(self, tmp_path):
        """Test load_image raises for non-existent path."""
        pytest.importorskip("PIL")

        exp = Experiment(
            id=1,
            prompt="test",
            path=str(tmp_path / "nonexistent.png"),
        )

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            exp.load_image()

    def test_load_image_directory_path(self, tmp_path):
        """Test load_image raises for directory path."""
        pytest.importorskip("PIL")

        # tmp_path is a directory
        exp = Experiment(
            id=1,
            prompt="test",
            path=str(tmp_path),
        )

        with pytest.raises(ValueError, match="not a regular file"):
            exp.load_image()

    def test_load_image_valid_path(self, tmp_path):
        """Test load_image works with valid path."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10), "red")
        img.save(img_path)

        exp = Experiment(
            id=1,
            prompt="test",
            path=str(img_path),
        )

        loaded = exp.load_image()
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (10, 10)


class TestAutoImageHash:
    """Tests for automatic image hash computation."""

    def test_log_auto_computes_hash(self, tmp_path):
        """Test that log() auto-computes hash when path provided."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10), "red")
        img.save(img_path)

        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test", path=str(img_path))

        experiments = tracker.find()
        assert len(experiments) == 1
        assert experiments[0].image_hash is not None
        assert experiments[0].image_hash.startswith("sha256:")

    def test_log_many_auto_computes_hash(self, tmp_path):
        """Test that log_many() auto-computes hashes."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create test images
        img_path1 = tmp_path / "test1.png"
        img_path2 = tmp_path / "test2.png"
        for path in [img_path1, img_path2]:
            img = Image.new("RGB", (10, 10), "red")
            img.save(path)

        tracker = Tracker(tmp_path / "test.db")
        tracker.log_many([
            {"prompt": "test1", "path": str(img_path1)},
            {"prompt": "test2", "path": str(img_path2)},
        ])

        experiments = tracker.find()
        assert len(experiments) == 2
        for exp in experiments:
            assert exp.image_hash is not None
            assert exp.image_hash.startswith("sha256:")


class TestDateFilters:
    """Tests for date-based filtering."""

    def test_find_created_after(self, tmp_path):
        """Test filtering by created_after."""
        from datetime import datetime, timedelta

        tracker = Tracker(tmp_path / "test.db")

        # Log two experiments
        tracker.log(prompt="old")
        tracker.log(prompt="new")

        # Filter for experiments after a past date
        past = datetime.now() - timedelta(days=1)
        experiments = tracker.find(created_after=past)
        assert len(experiments) == 2

        # Filter for experiments after now (should be empty)
        future = datetime.now() + timedelta(days=1)
        experiments = tracker.find(created_after=future)
        assert len(experiments) == 0

    def test_find_created_before(self, tmp_path):
        """Test filtering by created_before."""
        from datetime import datetime, timedelta

        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test")

        # Filter for experiments before future date
        future = datetime.now() + timedelta(days=1)
        experiments = tracker.find(created_before=future)
        assert len(experiments) == 1

        # Filter for experiments before past date (should be empty)
        past = datetime.now() - timedelta(days=1)
        experiments = tracker.find(created_before=past)
        assert len(experiments) == 0

    def test_find_date_range_with_strings(self, tmp_path):
        """Test date filtering with ISO string format."""
        from datetime import datetime, timedelta

        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test")

        # Use string dates
        past = (datetime.now() - timedelta(days=1)).isoformat()
        future = (datetime.now() + timedelta(days=1)).isoformat()

        experiments = tracker.find(created_after=past, created_before=future)
        assert len(experiments) == 1


class TestFilterEdgeCases:
    """Tests for filter edge cases."""

    def test_find_by_id(self, tmp_path):
        """Test filtering by experiment ID."""
        tracker = Tracker(tmp_path / "test.db")
        _id1 = tracker.log(prompt="first")  # noqa: F841
        id2 = tracker.log(prompt="second")
        _id3 = tracker.log(prompt="third")  # noqa: F841

        # Find specific experiment by ID
        experiments = tracker.find(id=id2)
        assert len(experiments) == 1
        assert experiments[0].id == id2
        assert experiments[0].prompt == "second"

        # Non-existent ID returns empty list
        experiments = tracker.find(id=9999)
        assert len(experiments) == 0

    def test_find_by_dimensions(self, tmp_path):
        """Test filtering by width and height."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test", width=512, height=768)
        tracker.log(prompt="test2", width=1024, height=1024)

        experiments = tracker.find(width=512)
        assert len(experiments) == 1
        assert experiments[0].width == 512

        experiments = tracker.find(height=1024)
        assert len(experiments) == 1
        assert experiments[0].height == 1024

    def test_find_by_prompt_startswith(self, tmp_path):
        """Test filtering by prompt prefix."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="cyberpunk city")
        tracker.log(prompt="cyber warrior")
        tracker.log(prompt="nature scene")

        experiments = tracker.find(prompt__startswith="cyber")
        assert len(experiments) == 2

    def test_find_with_special_characters(self, tmp_path):
        """Test filtering with SQL special characters."""
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(prompt="test%special_chars")
        tracker.log(prompt="normal prompt")

        # Should find exact match, not treat % as wildcard
        experiments = tracker.find(prompt__contains="%special")
        assert len(experiments) == 1
        assert experiments[0].prompt == "test%special_chars"
