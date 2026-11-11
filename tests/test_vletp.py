import pytest

from core.variable_leverage_product import VLETP


def make(leverage_target=None, short=False, lower=1.5, upper=3, start_leverage=2, start_nav=100):
    """NAV 100, leverage 2, band [1.5, 3]; mirrored to negative values for a short product."""
    sign = -1 if short else 1
    if isinstance(leverage_target, tuple):
        leverage_target = tuple(sign * target for target in leverage_target)
    elif leverage_target is not None:
        leverage_target = sign * leverage_target
    return VLETP(
        "DOWN" if short else "UP",
        start_nav=start_nav,
        start_leverage=sign * start_leverage,
        leverage_lower_bound=sign * lower,
        leverage_upper_bound=sign * upper,
        leverage_target=leverage_target,
    )


# --- construction -----------------------------------------------------------------


def test_initial_state_long(up_product):
    assert up_product.direction == 1
    assert up_product.nav == 100
    assert up_product.exposure == 200
    assert up_product.exposure_after == 200
    assert up_product.leverage == 2
    assert up_product.leverage_after == 2
    assert up_product.target_delta == 0
    assert up_product.actual_delta == 0
    assert not up_product.is_wiped_out


def test_initial_state_short(down_product):
    assert down_product.direction == -1
    assert down_product.nav == 100
    assert down_product.exposure == -200  # exposure is signed
    assert down_product.leverage == 2  # leverage is quoted as a magnitude
    assert (down_product.leverage_lower_bound, down_product.leverage_upper_bound) == (1, 3)


@pytest.mark.parametrize("lower, upper", [(-1.25, -4), (-4, -1.25)])
def test_short_band_accepted_in_either_order(lower, upper):
    product = VLETP("DOWN", 100, -2, lower, upper)
    assert product.direction == -1
    assert (product.leverage_lower_bound, product.leverage_upper_bound) == (1.25, 4)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start_nav": 0}, "start_nav"),
        ({"start_nav": -5}, "start_nav"),
        ({"leverage_lower_bound": 0}, "non-zero"),
        ({"leverage_upper_bound": 0}, "non-zero"),
        ({"leverage_lower_bound": -1}, "same sign"),
        ({"leverage_lower_bound": 3, "leverage_upper_bound": 1}, "must not exceed"),
        ({"start_leverage": -2}, "start_leverage"),
        ({"leverage_target": 0.5}, "leverage_target"),
        ({"leverage_target": 3.5}, "leverage_target"),
        ({"leverage_target": -2}, "leverage_target"),
        ({"leverage_target": (0.5, 2)}, "leverage_target"),
        ({"leverage_target": (2, 3.5)}, "leverage_target"),
        ({"leverage_target": (1.5, 2, 2.5)}, "leverage_target"),
        ({"leverage_target": (2,)}, "leverage_target"),
    ],
)
def test_invalid_arguments_rejected(kwargs, match):
    arguments = {"name": "UP", "start_nav": 100, "start_leverage": 2, "leverage_lower_bound": 1}
    arguments.update({"leverage_upper_bound": 3, **kwargs})
    with pytest.raises(ValueError, match=match):
        VLETP(**arguments)


def test_name_must_be_string():
    with pytest.raises(TypeError):
        VLETP(1, 100, 2, 1, 3)


@pytest.mark.parametrize(
    "short, given, stored, mode",
    [
        (False, None, None, "boundary"),
        (False, 2, 2, "target"),
        (False, (1.75, 2.5), (1.75, 2.5), "target"),
        (False, (2.5, 1.75), (1.75, 2.5), "target"),
        (True, 2, 2, "target"),
        (True, (1.75, 2.5), (1.75, 2.5), "target"),
        (True, (2.5, 1.75), (1.75, 2.5), "target"),
    ],
)
def test_leverage_target_stored_as_magnitudes(short, given, stored, mode):
    product = make(given, short=short)
    assert product.leverage_target == stored
    assert product.rebalancing_mode == mode


def test_target_on_the_boundary_is_allowed():
    assert make((1.5, 3)).leverage_target == (1.5, 3)


def test_from_basket():
    # 10 tokens at 10 = NAV 100; short 2 units of an underlying at 100 = exposure -200
    product = VLETP.from_basket(
        "DOWN",
        n_tokens=10,
        token_price=10,
        basket=-2,
        underlying_price=100,
        leverage_lower_bound=-1,
        leverage_upper_bound=-3,
        leverage_target=-2,
    )
    assert product.nav == 100
    assert product.exposure == -200
    assert product.leverage == 2
    assert product.leverage_target == 2


@pytest.mark.parametrize("start_leverage", [1, 3])
def test_start_leverage_on_the_boundary_needs_no_rebalance(start_leverage):
    product = VLETP("UP", 100, start_leverage, 1, 3)
    assert product.leverage == start_leverage
    assert not product.needs_rebalance()


@pytest.mark.parametrize(
    "start_leverage, lower, upper",
    [(5, 1, 3), (1.2, 1.5, 3), (-5, -1, -3), (-0.5, -1, -3), (-5, -3, -1)],
)
def test_start_leverage_outside_band_is_kept_as_given(start_leverage, lower, upper):
    product = VLETP("X", 100, start_leverage, lower, upper)
    assert product.leverage == abs(start_leverage)  # not clamped to the boundary
    assert product.leverage_after == abs(start_leverage)
    assert product.exposure == start_leverage * 100
    assert product.needs_rebalance()


@pytest.mark.parametrize(
    "start_leverage, leverage_target, expected_leverage",
    [
        (5, None, 3),
        (5, 2, 2),
        (5, (1.75, 2.5), 2.5),
        (1.2, None, 1.5),
        (1.2, 2, 2),
        (1.2, (1.75, 2.5), 1.75),
    ],
)
def test_start_leverage_outside_band_rebalances_on_first_step(start_leverage, leverage_target, expected_leverage):
    # Flat price, so the only reason to trade is the starting position itself
    product = make(leverage_target, start_leverage=start_leverage)
    executed = product.step(0)
    assert executed == pytest.approx((expected_leverage - start_leverage) * 100)
    assert product.leverage == pytest.approx(start_leverage)
    assert product.leverage_after == pytest.approx(expected_leverage)
    assert product.nav == pytest.approx(100)


def test_start_leverage_outside_band_marks_to_market_at_actual_leverage():
    # Leverage 5 on NAV 100: a 10% rise adds 50 to NAV and takes exposure from 500 to 550
    product = VLETP("UP", 100, 5, 1, 3)
    product.apply_return(0.1)
    assert product.nav == pytest.approx(150)
    assert product.exposure == pytest.approx(550)
    assert product.leverage == pytest.approx(550 / 150)


def test_from_basket_accepts_holdings_outside_the_band():
    # 5 units at 100 on a NAV of 100 is leverage 5
    product = VLETP.from_basket("UP", 10, 10, 5, 100, leverage_lower_bound=1, leverage_upper_bound=3)
    assert product.leverage == 5
    assert product.exposure == 500


# --- apply_return -------------------------------------------------------------------


def test_apply_return_long(up_product):
    up_product.apply_return(0.1)
    assert up_product.nav == pytest.approx(120)  # 100 * (1 + 2 * 0.1)
    assert up_product.exposure == pytest.approx(220)  # 200 * 1.1
    assert up_product.leverage == pytest.approx(220 / 120)
    assert up_product.leverage_after == up_product.leverage
    assert not up_product.needs_rebalance()


def test_apply_return_short(down_product):
    down_product.apply_return(0.1)
    assert down_product.nav == pytest.approx(80)  # 100 * (1 - 2 * 0.1)
    assert down_product.exposure == pytest.approx(-220)
    assert down_product.leverage == pytest.approx(2.75)
    assert not down_product.needs_rebalance()


def test_no_trade_while_inside_band(up_product):
    assert up_product.step(0.1) == 0
    assert up_product.target_delta == 0
    assert up_product.exposure_after == pytest.approx(220)
    assert up_product.rebalance_leverage() == up_product.leverage


def test_returns_compound_between_steps(up_product):
    up_product.step(0.1)
    up_product.step(0.1)
    leverage_after_first_step = 220 / 120
    assert up_product.nav == pytest.approx(120 * (1 + leverage_after_first_step * 0.1))
    assert up_product.exposure == pytest.approx(242)


# --- rebalancing --------------------------------------------------------------------

# Returns that push NAV 100 / leverage 2 outside the band [1.5, 3]:
#   long:  -0.4 -> NAV 20, exposure 120, leverage 6       +1.0 -> NAV 300, exposure 400, leverage 1.333
#   short: +0.4 -> NAV 20, exposure -280, leverage 14     -0.3 -> NAV 160, exposure -140, leverage 0.875
BREACHES = {
    (False, "upper"): (-0.4, 20, 120),
    (False, "lower"): (1.0, 300, 400),
    (True, "upper"): (0.4, 20, -280),
    (True, "lower"): (-0.3, 160, -140),
}


@pytest.mark.parametrize("short", [False, True], ids=["long", "short"])
@pytest.mark.parametrize(
    "leverage_target, breached, expected_leverage",
    [
        (None, "lower", 1.5),  # nearest boundary
        (None, "upper", 3),
        (2, "lower", 2),  # single number: always that target
        (2, "upper", 2),
        ((1.75, 2.5), "lower", 1.75),  # pair: nearest of the two
        ((1.75, 2.5), "upper", 2.5),
        ((2.5, 1.75), "lower", 1.75),  # order of the pair does not matter
        ((2.5, 1.75), "upper", 2.5),
    ],
)
def test_rebalance_target(short, leverage_target, breached, expected_leverage):
    ret, nav, exposure = BREACHES[(short, breached)]
    direction = -1 if short else 1
    product = make(leverage_target, short=short)

    product.apply_return(ret)
    assert product.nav == pytest.approx(nav)
    assert product.exposure == pytest.approx(exposure)
    assert product.needs_rebalance()
    assert product.rebalance_leverage() == expected_leverage

    expected_delta = direction * nav * expected_leverage - exposure
    assert product.calculate_rebalance() == pytest.approx(expected_delta)
    assert product.execute_rebalance() == pytest.approx(expected_delta)
    assert product.leverage == pytest.approx(abs(exposure) / nav)  # pre-rebalance leverage kept
    assert product.leverage_after == pytest.approx(expected_leverage)
    assert product.exposure_after == pytest.approx(direction * nav * expected_leverage)


def test_partial_fill_leaves_leverage_outside_band(up_product):
    # -0.4: NAV 20, exposure 120, leverage 6; wants to sell 60 to reach leverage 3
    up_product.apply_return(-0.4)
    assert up_product.calculate_rebalance() == pytest.approx(-60)
    assert up_product.execute_rebalance(scale=0.5) == pytest.approx(-30)
    assert up_product.target_delta == pytest.approx(-60)
    assert up_product.actual_delta == pytest.approx(-30)
    assert up_product.exposure_after == pytest.approx(90)
    assert up_product.leverage_after == pytest.approx(4.5)


def test_step_matches_the_three_separate_calls():
    stepped, manual = make(2), make(2)
    executed = stepped.step(-0.4, scale=0.5)
    manual.apply_return(-0.4)
    manual.calculate_rebalance()
    assert manual.execute_rebalance(0.5) == executed
    assert stepped.state() == manual.state()


def test_state_keys(up_product):
    assert set(up_product.state()) == {"v", "x", "x_star", "lam", "lam_star", "target_delta", "actual_delta"}


# --- wipe-out -----------------------------------------------------------------------


@pytest.mark.parametrize("ret", [-0.5, -0.8])
def test_wipe_out_liquidates_the_position(up_product, ret):
    up_product.apply_return(ret)
    assert up_product.is_wiped_out
    assert up_product.nav == 0
    assert up_product.exposure == 0
    assert up_product.leverage == 0
    assert up_product.target_delta == pytest.approx(-200 * (1 + ret))  # sell what is left
    assert not up_product.needs_rebalance()
    assert up_product.calculate_rebalance() == up_product.target_delta


def test_short_product_wiped_out_by_a_rally(down_product):
    down_product.apply_return(0.5)
    assert down_product.is_wiped_out
    assert down_product.target_delta == pytest.approx(300)  # buy back the short


def test_wiped_out_product_stays_dead(up_product):
    up_product.step(-0.5)
    assert up_product.step(0.9) == 0
    assert up_product.is_wiped_out
    assert up_product.nav == 0
    assert up_product.exposure_after == 0


def test_reset_restores_initial_state(up_product):
    initial = up_product.state()
    up_product.step(-0.5)
    up_product.reset()
    assert up_product.state() == initial
    assert not up_product.is_wiped_out
