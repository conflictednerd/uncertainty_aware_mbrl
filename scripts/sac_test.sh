# python -m src.main exp_name="button_press_wall_baseline" policy="button_press_wall"
python -m src.main exp_name="button_press_baseline" policy="button_press"
python -m src.main exp_name="button_press_topdown_baseline" policy="button_press_topdown"

python -m src.main exp_name="door_open_baseline" policy="door_open"
python -m src.main exp_name="door_close_baseline" policy="door_close"
python -m src.main exp_name="door_lock_baseline" policy="door_lock"
python -m src.main exp_name="door_unlock_baseline" policy="door_unlock"

python -m src.main exp_name="drawer_open_baseline" policy="drawer_open"
python -m src.main exp_name="drawer_close_baseline" policy="drawer_close"

python -m src.main exp_name="basketball_baseline" policy="basketball"
python -m src.main exp_name="pick_place_baseline" policy="pick_place"