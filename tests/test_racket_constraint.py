import unittest
import torch

from src.racket.player_constraint import allowed_region,decode_player


class PlayerConstraintTests(unittest.TestCase):
    def test_opponent_peak_is_replaced_by_valid_player_peak_not_clamped(self):
        logits=torch.full((1,1,100,100),-20.)
        logits[0,0,10,50]=15. # opponent above owner region
        logits[0,0,60,70]=10. # actual racket candidate outside body box
        boxes=[[40,40,60,80]];crops=[[0,0,100]]
        point,mass,valid=decode_player(logits,boxes,crops,policy='expanded_box')
        self.assertTrue(bool(valid[0]))
        torch.testing.assert_close(point,torch.tensor([[.705,.605]]),atol=.001,rtol=0)
        self.assertLess(float(mass[0]),.01) # unrenormalized original confidence
        tight=allowed_region(logits,boxes,crops,policy='body_box')
        self.assertFalse(bool(tight[0,60,70]))

    def test_missing_wrist_pose_falls_back_to_expanded_body_region(self):
        logits=torch.zeros(1,1,32,32);boxes=[[40,40,60,80]];crops=[[0,0,100]]
        base=allowed_region(logits,boxes,crops,policy='expanded_box')
        fallback=allowed_region(logits,boxes,crops,torch.zeros(1,17,3),'wrist_region')
        self.assertTrue(torch.equal(base,fallback))

    def test_region_outside_crop_returns_invalid_not_edge_prediction(self):
        logits=torch.zeros(1,1,32,32)
        _,_,valid=decode_player(logits,[[1000,1000,1020,1050]],[[0,0,100]])
        self.assertFalse(bool(valid[0]))

    def test_constraint_uses_original_pixel_geometry_after_crop_offset(self):
        logits=torch.full((1,1,100,100),-20.)
        logits[0,0,10,50]=20.;logits[0,0,60,70]=10.
        # Same geometry under +200,+300 translation and 2x scale.
        point,_,_=decode_player(logits,[[280,380,320,460]],[[200,300,200]],policy='expanded_box')
        torch.testing.assert_close(point,torch.tensor([[.705,.605]]),atol=.001,rtol=0)


if __name__=='__main__':unittest.main()
