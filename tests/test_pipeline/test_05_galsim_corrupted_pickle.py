import logging

from mejiro.pipeline import _05_galsim


def test_get_image_skips_corrupted_pickle(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING)

    class DummyPipeline:
        instrument_name = 'jwst'
        output_dir = str(tmp_path)

    imaging_config = {
        'exposure_time': 1.0,
        'engine': 'galsim',
        'engine_params': {},
        'serialization': 'lightweight',
    }
    corrupted = tmp_path / 'SyntheticImage_test_F129.pkl'
    corrupted.write_bytes(b'')

    def raise_eof(path):
        raise EOFError('ran out of input')

    monkeypatch.setattr(_05_galsim.util, 'unpickle', raise_eof)

    result = _05_galsim.get_image((DummyPipeline(), imaging_config, str(corrupted)))

    assert result is None
    assert any(
        'Skipping corrupted SyntheticImage file' in record.message
        for record in caplog.records
    )
