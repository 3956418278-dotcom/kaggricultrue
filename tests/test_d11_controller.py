"""Contract-focused checks for the D11+ programme controller."""
from dataclasses import replace
import unittest

from kaggle_environments import make

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.intraday import solve_intraday
from src.kaggriculture_agent.market import (
    REVEAL_DAYS, known_demand_events, next_reveal, optimize_sales,
    opponent_pressure, sell_revenue,
)
from src.kaggriculture_agent.programme import Programme, ProgrammeEvent
from src.kaggriculture_agent.simulation import simulate_programme
from src.kaggriculture_agent.state import reconstruct


class D11ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        obs=make("kaggriculture",configuration={"seed":11}).reset(2)[0].observation
        cls.state=replace(reconstruct(obs),step=10*24,day=10,turn=0)

    def test_canonical_state_has_private_and_visible_exact_assets(self):
        state=self.state
        self.assertEqual(state.turn,0)
        self.assertEqual(state.own.worker_positions,(state.workers[0].position,))
        self.assertEqual(state.own.shed_inventory,state.shed)
        self.assertEqual(len(state.own.usable_tiles),25)
        self.assertEqual(len(state.opp.usable_tiles),25)
        self.assertTrue(hasattr(state.market,"inventory"))
        self.assertTrue(hasattr(state.market,"price"))

    def test_reveal_and_demand_are_discrete_events(self):
        self.assertEqual(REVEAL_DAYS,(4,7,10,13,16,19,22,25))
        self.assertEqual(next_reveal(10),13)
        self.assertEqual(next_reveal(25),31)
        events=known_demand_events(self.state)
        self.assertEqual(events["WHEAT"][:2],((240,1),(264,1)))
        self.assertEqual(events["FERTILIZER"],())

    def test_sequential_sale_uses_each_updated_inventory(self):
        inventory=9998
        expected=(rules.market_price("MILK",inventory)+
                  rules.market_price("MILK",inventory+1)+
                  rules.market_price("MILK",inventory+2))
        self.assertEqual(sell_revenue("MILK",3,inventory),expected)

    def test_sale_dp_does_not_put_unsold_stock_into_market(self):
        state=replace(self.state,own=replace(self.state.own,
            shed_inventory={**self.state.shed,"MILK":2}))
        result=optimize_sales(state,{},known_demand_events(state),opponent_pressure(state))
        sold=sum(amounts.get("MILK",0) for amounts in result.planned_sale.values())
        self.assertEqual(sold,2)
        first=min(result.market_inventory)
        self.assertLessEqual(result.market_inventory[first]["MILK"],state.market.inventory["MILK"]+2)

    def test_intraday_keeps_macro_asset_and_tile_frozen(self):
        event=ProgrammeEvent(240,10,"water","WATER",(0,0),"crop",action=("WATER",),deadline=263)
        plan=Programme(240,10,(),events=(event,))
        solved=solve_intraday(self.state,plan)
        self.assertEqual(solved.assets,plan.assets)
        self.assertEqual(solved.events_at(240)[0].tile,(0,0))
        self.assertTrue(all(route.actions for route in solved.routes))
        covered={step for route in solved.routes if route.worker==0 for step in route.actions}
        self.assertEqual(covered,set(range(240,719)))

    def test_simulator_keeps_production_and_market_inventory_separate(self):
        plan=Programme(240,10,())
        solved=solve_intraday(self.state,plan)
        result=simulate_programme(self.state,solved,pressure_events={p:() for p in rules.PRODUCTS})
        self.assertTrue(result.feasible)
        self.assertEqual(result.terminal_cash,self.state.money)


if __name__ == "__main__":
    unittest.main()
