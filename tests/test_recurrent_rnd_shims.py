from datenwissenschaften import recurrent_rnd
from datenwissenschaften.recurrent_rnd import model as recurrent_rnd_model
from datenwissenschaften.rnd import model as rnd_model


def test_package_reexports_match_the_rnd_package():
    assert recurrent_rnd.AdaptiveRecurrentRNDModel is rnd_model.AdaptiveRecurrentRNDModel
    assert recurrent_rnd.AdaptiveRecurrentRNDPPO is rnd_model.AdaptiveRecurrentRNDPPO
    assert recurrent_rnd.build_adaptive_recurrent_rnd_ppo is rnd_model.build_adaptive_recurrent_rnd_ppo
    assert recurrent_rnd.__all__ == [
        "AdaptiveRecurrentRNDModel",
        "AdaptiveRecurrentRNDPPO",
        "build_adaptive_recurrent_rnd_ppo",
    ]


def test_model_module_reexports_match_the_rnd_model_module():
    assert recurrent_rnd_model.AdaptiveRecurrentRNDModel is rnd_model.AdaptiveRecurrentRNDModel
    assert recurrent_rnd_model.AdaptiveRecurrentRNDPPO is rnd_model.AdaptiveRecurrentRNDPPO
    assert recurrent_rnd_model.build_adaptive_recurrent_rnd_ppo is rnd_model.build_adaptive_recurrent_rnd_ppo
