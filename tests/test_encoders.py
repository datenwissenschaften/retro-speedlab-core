import threading

from datenwissenschaften.speedrun.encoders import encoder_slot

WAIT_SECONDS = 2.0


def test_an_encode_waits_while_every_slot_is_taken():
    entered = threading.Event()

    def encode() -> None:
        with encoder_slot(1):
            entered.set()

    with encoder_slot(1):
        threading.Thread(target=encode).start()
        assert not entered.wait(WAIT_SECONDS / 4)
    assert entered.wait(WAIT_SECONDS)


def test_encodes_run_side_by_side_while_slots_are_free():
    with encoder_slot(2), encoder_slot(2):
        pass
