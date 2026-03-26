# Critic Update Radius Threshold Audit

- checkpoint: `checkpoints\dense_policy_joint_reward_aligned_credit_experiment_20260321_170716\policy_ratio_hierarchical_actor_joint_reward_aligned_credit_20260321_172521_442663\best_model.pt`
- captured minibatch: epoch `0` / update_epoch `2` / minibatch `8`
- target_update_norms: `[1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001]`
- interpretation: A semantic stability radius exists if small matched pure-gradient updates keep current, held-out, and probe roughly aligned, while larger update norms start producing 'current better, global worse'.

## Gradient Alignment

```json
{
  "current_vs_global_cosine": 0.8179006510940898,
  "current_vs_blended_cosine": 0.8998173965551086,
  "global_vs_blended_cosine": 0.9864117741965149,
  "grad_norm_current": 2.131699207039702,
  "grad_norm_global": 1.882236064373162,
  "grad_norm_blended": 1.8636237215606646
}
```

## Radius Thresholds

```json
{
  "current_grad": {
    "first_loss_triad_failure_norm": null,
    "first_probe_target_pearson_drop_norm": 1e-05,
    "per_norm": {
      "1e-05": {
        "current_batch": {
          "critic_loss_delta": -4.267692565917969e-05,
          "target_pearson_delta": 1.2516975402832031e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -2.446770668029785e-05,
          "target_pearson_delta": 1.3709068298339844e-06
        },
        "probe": {
          "critic_loss_delta": -3.74913215637207e-05,
          "target_pearson_delta": -2.473592758178711e-06
        }
      },
      "2e-05": {
        "current_batch": {
          "critic_loss_delta": -8.52048397064209e-05,
          "target_pearson_delta": 2.485513687133789e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -4.89354133605957e-05,
          "target_pearson_delta": 2.7418136596679688e-06
        },
        "probe": {
          "critic_loss_delta": -7.480382919311523e-05,
          "target_pearson_delta": -4.887580871582031e-06
        }
      },
      "5e-05": {
        "current_batch": {
          "critic_loss_delta": -0.00021311640739440918,
          "target_pearson_delta": 6.22868537902832e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -0.00012224912643432617,
          "target_pearson_delta": 6.854534149169922e-06
        },
        "probe": {
          "critic_loss_delta": -0.0001869797706604004,
          "target_pearson_delta": -1.2308359146118164e-05
        }
      },
      "1e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0004261434078216553,
          "target_pearson_delta": 0.0001246929168701172
        },
        "held_out_batch": {
          "critic_loss_delta": -0.00024431943893432617,
          "target_pearson_delta": 1.4066696166992188e-05
        },
        "probe": {
          "critic_loss_delta": -0.0003738999366760254,
          "target_pearson_delta": -2.4616718292236328e-05
        }
      },
      "2e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0008519589900970459,
          "target_pearson_delta": 0.00024956464767456055
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0004884302616119385,
          "target_pearson_delta": 2.8133392333984375e-05
        },
        "probe": {
          "critic_loss_delta": -0.0007475018501281738,
          "target_pearson_delta": -4.935264587402344e-05
        }
      },
      "5e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0021276772022247314,
          "target_pearson_delta": 0.0006251335144042969
        },
        "held_out_batch": {
          "critic_loss_delta": -0.001219630241394043,
          "target_pearson_delta": 7.039308547973633e-05
        },
        "probe": {
          "critic_loss_delta": -0.0018677115440368652,
          "target_pearson_delta": -0.00012350082397460938
        }
      },
      "1e-03": {
        "current_batch": {
          "critic_loss_delta": -0.004247456789016724,
          "target_pearson_delta": 0.001253664493560791
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0024341344833374023,
          "target_pearson_delta": 0.0001405477523803711
        },
        "probe": {
          "critic_loss_delta": -0.003732144832611084,
          "target_pearson_delta": -0.0002474188804626465
        }
      }
    }
  },
  "global_grad": {
    "first_loss_triad_failure_norm": null,
    "first_probe_target_pearson_drop_norm": 1e-05,
    "per_norm": {
      "1e-05": {
        "current_batch": {
          "critic_loss_delta": -3.4928321838378906e-05,
          "target_pearson_delta": 2.1457672119140625e-06
        },
        "held_out_batch": {
          "critic_loss_delta": -3.8683414459228516e-05,
          "target_pearson_delta": 2.8014183044433594e-06
        },
        "probe": {
          "critic_loss_delta": -5.054473876953125e-05,
          "target_pearson_delta": -3.248453140258789e-06
        }
      },
      "2e-05": {
        "current_batch": {
          "critic_loss_delta": -6.970763206481934e-05,
          "target_pearson_delta": 4.112720489501953e-06
        },
        "held_out_batch": {
          "critic_loss_delta": -7.733702659606934e-05,
          "target_pearson_delta": 5.662441253662109e-06
        },
        "probe": {
          "critic_loss_delta": -0.00010085105895996094,
          "target_pearson_delta": -6.556510925292969e-06
        }
      },
      "5e-05": {
        "current_batch": {
          "critic_loss_delta": -0.00017431378364562988,
          "target_pearson_delta": 1.0311603546142578e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0001933276653289795,
          "target_pearson_delta": 1.430511474609375e-05
        },
        "probe": {
          "critic_loss_delta": -0.0002524256706237793,
          "target_pearson_delta": -1.633167266845703e-05
        }
      },
      "1e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0003484785556793213,
          "target_pearson_delta": 2.0503997802734375e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0003865957260131836,
          "target_pearson_delta": 2.849102020263672e-05
        },
        "probe": {
          "critic_loss_delta": -0.0005047321319580078,
          "target_pearson_delta": -3.2573938369750977e-05
        }
      },
      "2e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0006968677043914795,
          "target_pearson_delta": 4.112720489501953e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -0.000773012638092041,
          "target_pearson_delta": 5.704164505004883e-05
        },
        "probe": {
          "critic_loss_delta": -0.0010089874267578125,
          "target_pearson_delta": -6.526708602905273e-05
        }
      },
      "5e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0017405450344085693,
          "target_pearson_delta": 0.00010329484939575195
        },
        "held_out_batch": {
          "critic_loss_delta": -0.001931607723236084,
          "target_pearson_delta": 0.0001423358917236328
        },
        "probe": {
          "critic_loss_delta": -0.0025217533111572266,
          "target_pearson_delta": -0.0001634657382965088
        }
      },
      "1e-03": {
        "current_batch": {
          "critic_loss_delta": -0.003475278615951538,
          "target_pearson_delta": 0.0002079606056213379
        },
        "held_out_batch": {
          "critic_loss_delta": -0.003860384225845337,
          "target_pearson_delta": 0.0002837181091308594
        },
        "probe": {
          "critic_loss_delta": -0.005040407180786133,
          "target_pearson_delta": -0.00032764673233032227
        }
      }
    }
  },
  "blended_grad": {
    "first_loss_triad_failure_norm": null,
    "first_probe_target_pearson_drop_norm": 1e-05,
    "per_norm": {
      "1e-05": {
        "current_batch": {
          "critic_loss_delta": -3.832578659057617e-05,
          "target_pearson_delta": 5.304813385009766e-06
        },
        "held_out_batch": {
          "critic_loss_delta": -3.6150217056274414e-05,
          "target_pearson_delta": 2.6226043701171875e-06
        },
        "probe": {
          "critic_loss_delta": -4.774332046508789e-05,
          "target_pearson_delta": -3.2782554626464844e-06
        }
      },
      "2e-05": {
        "current_batch": {
          "critic_loss_delta": -7.668137550354004e-05,
          "target_pearson_delta": 1.049041748046875e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -7.236003875732422e-05,
          "target_pearson_delta": 5.245208740234375e-06
        },
        "probe": {
          "critic_loss_delta": -9.584426879882812e-05,
          "target_pearson_delta": -6.496906280517578e-06
        }
      },
      "5e-05": {
        "current_batch": {
          "critic_loss_delta": -0.00019165873527526855,
          "target_pearson_delta": 2.6226043701171875e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -0.00018084049224853516,
          "target_pearson_delta": 1.329183578491211e-05
        },
        "probe": {
          "critic_loss_delta": -0.00023955106735229492,
          "target_pearson_delta": -1.6242265701293945e-05
        }
      },
      "1e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0003834962844848633,
          "target_pearson_delta": 5.257129669189453e-05
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0003618299961090088,
          "target_pearson_delta": 2.676248550415039e-05
        },
        "probe": {
          "critic_loss_delta": -0.00047904253005981445,
          "target_pearson_delta": -3.236532211303711e-05
        }
      },
      "2e-04": {
        "current_batch": {
          "critic_loss_delta": -0.000766754150390625,
          "target_pearson_delta": 0.00010532140731811523
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0007233917713165283,
          "target_pearson_delta": 5.340576171875e-05
        },
        "probe": {
          "critic_loss_delta": -0.0009580254554748535,
          "target_pearson_delta": -6.464123725891113e-05
        }
      },
      "5e-04": {
        "current_batch": {
          "critic_loss_delta": -0.0019149482250213623,
          "target_pearson_delta": 0.00026357173919677734
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0018073320388793945,
          "target_pearson_delta": 0.00013333559036254883
        },
        "probe": {
          "critic_loss_delta": -0.0023941993713378906,
          "target_pearson_delta": -0.00016185641288757324
        }
      },
      "1e-03": {
        "current_batch": {
          "critic_loss_delta": -0.0038234293460845947,
          "target_pearson_delta": 0.000528872013092041
        },
        "held_out_batch": {
          "critic_loss_delta": -0.0036107301712036133,
          "target_pearson_delta": 0.0002658367156982422
        },
        "probe": {
          "critic_loss_delta": -0.004785060882568359,
          "target_pearson_delta": -0.0003243386745452881
        }
      }
    }
  }
}
```