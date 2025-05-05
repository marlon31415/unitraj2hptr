import unittest
import h5py

from convert import (
    convert_unitraj_to_hptr_agent,
    convert_unitraj_to_hptr_history_agent,
    convert_unitraj_to_hptr_map,
    convert_unitraj_to_hptr_tl,
    convert_unitraj_to_hptr_history_tl,
)

n_agent = 64
n_agent_no_sim = 256
n_steps = 91
n_steps_history = 11
n_pl = 1024
n_pl_node = 20


class TestConvertUnitrajToHptr(unittest.TestCase):
    def setUp(self):
        unitraj_data_dir = (
            "/mrtstorage/datasets_tmp/nuscenes_unitraj/train/nuscenes_scenarionet/0.h5"
        )

        with h5py.File(unitraj_data_dir, "r") as f:
            groups = list(f.keys())
            group = groups[0]
            self.data = {k: v[()] for k, v in f[group].items()}

    def test_convert_unitraj_to_hptr_agent(self):
        hptr_data = {}
        convert_unitraj_to_hptr_agent(self.data, hptr_data)

        self.assertEqual(hptr_data["agent/acc"].shape, (n_steps, n_agent, 1))
        self.assertEqual(hptr_data["agent/cmd"].shape, (n_agent, 8))
        self.assertEqual(hptr_data["agent/dest"].shape, (n_agent,))
        self.assertEqual(hptr_data["agent/goal"].shape, (n_agent, 4))
        self.assertEqual(hptr_data["agent/object_id"].shape, (n_agent,))
        self.assertEqual(hptr_data["agent/pos"].shape, (n_steps, n_agent, 2))
        self.assertEqual(hptr_data["agent/role"].shape, (n_agent, 3))
        self.assertEqual(hptr_data["agent/size"].shape, (n_agent, 3))
        self.assertEqual(hptr_data["agent/spd"].shape, (n_steps, n_agent, 1))
        self.assertEqual(hptr_data["agent/type"].shape, (n_agent, 3))
        self.assertEqual(hptr_data["agent/valid"].shape, (n_steps, n_agent))
        self.assertEqual(hptr_data["agent/vel"].shape, (n_steps, n_agent, 2))
        self.assertEqual(hptr_data["agent/yaw_bbox"].shape, (n_steps, n_agent, 1))
        self.assertEqual(hptr_data["agent/yaw_rate"].shape, (n_steps, n_agent, 1))
        self.assertEqual(hptr_data["agent_no_sim/object_id"].shape, (n_agent_no_sim,))
        self.assertEqual(
            hptr_data["agent_no_sim/pos"].shape, (n_steps, n_agent_no_sim, 2)
        )
        self.assertEqual(hptr_data["agent_no_sim/size"].shape, (n_agent_no_sim, 3))
        self.assertEqual(
            hptr_data["agent_no_sim/spd"].shape, (n_steps, n_agent_no_sim, 1)
        )
        self.assertEqual(hptr_data["agent_no_sim/type"].shape, (n_agent_no_sim, 3))
        self.assertEqual(
            hptr_data["agent_no_sim/valid"].shape, (n_steps, n_agent_no_sim)
        )
        self.assertEqual(
            hptr_data["agent_no_sim/vel"].shape, (n_steps, n_agent_no_sim, 2)
        )
        self.assertEqual(
            hptr_data["agent_no_sim/yaw_bbox"].shape, (n_steps, n_agent_no_sim, 1)
        )

        self.assertEqual(hptr_data["agent/valid"].dtype, "bool")
        self.assertEqual(hptr_data["agent/type"].dtype, "bool")
        self.assertEqual(hptr_data["agent/cmd"].dtype, "bool")
        self.assertEqual(hptr_data["agent/role"].dtype, "bool")
        self.assertEqual(hptr_data["agent_no_sim/valid"].dtype, "bool")
        self.assertEqual(hptr_data["agent_no_sim/type"].dtype, "bool")

        # check if all agents (with a role: ego, interaction, predict)
        # have at least one valid entry
        self.assertTrue(
            hptr_data["agent/valid"]
            .transpose(1, 0)[hptr_data["agent/role"].any(-1)]
            .any(-1)
            .all()
        )

    def test_convert_unitraj_to_hptr_history_agent(self):
        hptr_data = {}
        convert_unitraj_to_hptr_history_agent(self.data, hptr_data)

        self.assertEqual(
            hptr_data["history/agent/acc"].shape, (n_steps_history, n_agent, 1)
        )
        self.assertEqual(hptr_data["history/agent/object_id"].shape, (n_agent,))
        self.assertEqual(
            hptr_data["history/agent/pos"].shape, (n_steps_history, n_agent, 2)
        )
        self.assertEqual(hptr_data["history/agent/role"].shape, (n_agent, 3))
        self.assertEqual(hptr_data["history/agent/size"].shape, (n_agent, 3))
        self.assertEqual(
            hptr_data["history/agent/spd"].shape, (n_steps_history, n_agent, 1)
        )
        self.assertEqual(hptr_data["history/agent/type"].shape, (n_agent, 3))
        self.assertEqual(
            hptr_data["history/agent/valid"].shape, (n_steps_history, n_agent)
        )
        self.assertEqual(
            hptr_data["history/agent/vel"].shape, (n_steps_history, n_agent, 2)
        )
        self.assertEqual(
            hptr_data["history/agent/yaw_bbox"].shape, (n_steps_history, n_agent, 1)
        )
        self.assertEqual(
            hptr_data["history/agent/yaw_rate"].shape, (n_steps_history, n_agent, 1)
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/object_id"].shape, (n_agent_no_sim,)
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/pos"].shape,
            (n_steps_history, n_agent_no_sim, 2),
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/size"].shape, (n_agent_no_sim, 3)
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/spd"].shape,
            (n_steps_history, n_agent_no_sim, 1),
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/type"].shape, (n_agent_no_sim, 3)
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/valid"].shape,
            (n_steps_history, n_agent_no_sim),
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/vel"].shape,
            (n_steps_history, n_agent_no_sim, 2),
        )
        self.assertEqual(
            hptr_data["history/agent_no_sim/yaw_bbox"].shape,
            (n_steps_history, n_agent_no_sim, 1),
        )

        self.assertEqual(hptr_data["history/agent/valid"].dtype, "bool")
        self.assertEqual(hptr_data["history/agent/type"].dtype, "bool")
        self.assertEqual(hptr_data["history/agent/role"].dtype, "bool")
        self.assertEqual(hptr_data["history/agent_no_sim/valid"].dtype, "bool")
        self.assertEqual(hptr_data["history/agent_no_sim/type"].dtype, "bool")

    def test_convert_unitraj_to_hptr_map(self):
        hptr_data = {}
        convert_unitraj_to_hptr_map(self.data, hptr_data)

        self.assertEqual(hptr_data["map/boundary"].shape, (4,))
        self.assertEqual(hptr_data["map/dir"].shape, (n_pl, n_pl_node, 2))
        self.assertEqual(hptr_data["map/id"].shape, (n_pl,))
        self.assertEqual(hptr_data["map/pos"].shape, (n_pl, n_pl_node, 2))
        self.assertEqual(hptr_data["map/type"].shape, (n_pl, 11))
        self.assertEqual(hptr_data["map/valid"].shape, (n_pl, n_pl_node))

        self.assertEqual(hptr_data["map/valid"].dtype, "bool")
        self.assertEqual(hptr_data["map/type"].dtype, "bool")

    def test_convert_unitraj_to_hptr_tl(self):
        hptr_data = {}
        convert_unitraj_to_hptr_tl(self.data, hptr_data)

        self.assertEqual(hptr_data["tl_lane/idx"].shape, (n_steps, 100))
        self.assertEqual(hptr_data["tl_lane/state"].shape, (n_steps, 100, 5))
        self.assertEqual(hptr_data["tl_lane/valid"].shape, (n_steps, 100))
        self.assertEqual(hptr_data["tl_stop/dir"].shape, (n_steps, 40, 2))
        self.assertEqual(hptr_data["tl_stop/pos"].shape, (n_steps, 40, 2))
        self.assertEqual(hptr_data["tl_stop/state"].shape, (n_steps, 40, 5))
        self.assertEqual(hptr_data["tl_stop/valid"].shape, (n_steps, 40))

        self.assertEqual(hptr_data["tl_lane/valid"].dtype, "bool")
        self.assertEqual(hptr_data["tl_lane/state"].dtype, "bool")
        self.assertEqual(hptr_data["tl_stop/valid"].dtype, "bool")
        self.assertEqual(hptr_data["tl_stop/state"].dtype, "bool")

    def test_convert_unitraj_to_hptr_history_tl(self):
        hptr_data = {}
        convert_unitraj_to_hptr_history_tl(self.data, hptr_data)

        self.assertEqual(hptr_data["history/tl_lane/idx"].shape, (n_steps_history, 100))
        self.assertEqual(
            hptr_data["history/tl_lane/state"].shape, (n_steps_history, 100, 5)
        )
        self.assertEqual(
            hptr_data["history/tl_lane/valid"].shape, (n_steps_history, 100)
        )
        self.assertEqual(
            hptr_data["history/tl_stop/dir"].shape, (n_steps_history, 40, 2)
        )
        self.assertEqual(
            hptr_data["history/tl_stop/pos"].shape, (n_steps_history, 40, 2)
        )
        self.assertEqual(
            hptr_data["history/tl_stop/state"].shape, (n_steps_history, 40, 5)
        )
        self.assertEqual(
            hptr_data["history/tl_stop/valid"].shape, (n_steps_history, 40)
        )


if __name__ == "__main__":
    unittest.main()
