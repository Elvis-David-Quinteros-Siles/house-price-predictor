import pytest

from notify import EVENTS, PREFIX, compose, main, parse_args


@pytest.mark.parametrize("event", sorted(EVENTS))
def test_every_event_has_subject_and_meaning(event):
    subject, body = compose(event)
    assert subject.startswith(PREFIX) and body.strip()


def test_detail_summary_and_build_go_in_the_mail():
    subject, body = compose("promoted", "v5 en producción", "https://jenkins/job/house-price-ct/1/",
                            "@champion: v5")
    assert subject.endswith(": v5 en producción")
    assert "@champion: v5" in body and "https://jenkins/job/house-price-ct/1/" in body


def test_without_smtp_nothing_is_sent(monkeypatch, capsys):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.setattr("notify.registry_summary", lambda config: "")
    monkeypatch.setattr("sys.argv", ["notify.py", "failed", "--detail", "exit 1"])
    assert main(parse_args()) == 0
    assert "not sent" in capsys.readouterr().out
