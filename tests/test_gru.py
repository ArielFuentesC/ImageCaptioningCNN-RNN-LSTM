import pytest
import torch

from lstm_gru.gru_error import GRUCell

HIDDEN_STATE_SIZE, INPUT_SIZE = 7, 3
GRU = GRUCell(hidden_state_size=HIDDEN_STATE_SIZE, input_size=INPUT_SIZE)


def turn_off_all_the_gates():
    # Turn off all the gates by setting all the weights to 0 and biases to a very low -ve value
    # Gates will be off because sigmoid of low -ve value is 0
    for name, w in GRU.named_parameters():
        if 'weight' in name:
            w.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))
        if 'bias' in name:
            w.data = torch.nn.Parameter(-1000. * torch.ones(1, HIDDEN_STATE_SIZE))
    # The hidden state update uses tanh so it'll be off when both weight and biases are 0 so set bias to 0
    GRU.bias.data = torch.nn.Parameter(torch.zeros(1, HIDDEN_STATE_SIZE))


@pytest.mark.parametrize('x, hidden_state', [
    ([0.1] * INPUT_SIZE, [7.] * HIDDEN_STATE_SIZE),
    ([0.5] * INPUT_SIZE, [8.] * HIDDEN_STATE_SIZE),
    ([1.] * INPUT_SIZE, [9.] * HIDDEN_STATE_SIZE),
    ([1.5] * INPUT_SIZE, [10.] * HIDDEN_STATE_SIZE),
    ([2.] * INPUT_SIZE, [11.] * HIDDEN_STATE_SIZE),
    ([2.5] * INPUT_SIZE, [12.] * HIDDEN_STATE_SIZE)
])
def test_gru_update_gate_and_hidden_state(x, hidden_state):
    """
    This test tries to test the update gate equation.
    It's not possible to isolate a GRU equation without having a separate instance variable for each equation so
    this test will fail even when the error lies outside the update equation.

    The basic idea is same as the one for the LSTM unit tests. You can check the doc strings there.

    :param x: The input to the GRU
    :param hidden_state: The hidden state to be provided as the input to the GRU
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Turn of all the gates of GRU
    turn_off_all_the_gates()

    # Set update gate's bias to 0
    GRU.bias_u.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    # Now turn on the update gate via weights
    GRU.weight_u.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(1000.))

    # Run the forward pass to get the hidden state
    new_hidden_state = GRU(x, hidden_state.clone()).squeeze()

    assert torch.isclose(hidden_state.squeeze(), new_hidden_state).all(),\
        'Update gate test failed when update gate was open via weights and other things were closed.'

    # Turn off the update gate (via biases)
    GRU.bias_u.data = torch.nn.Parameter(-1000. * torch.ones(1, HIDDEN_STATE_SIZE))
    GRU.weight_u.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))

    # Run the forward pass to get the hidden state
    new_hidden_state = GRU(x, hidden_state.clone()).squeeze()

    assert torch.isclose(torch.zeros_like(new_hidden_state), new_hidden_state).all(), \
        'Update gate test failed when update gate was CLOSED via BIAS.'

    # Set new hidden state's weights to 1 and bias to 0 so that the output is the new hidden state
    # which should be tanh(sum(x))
    GRU.weight.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(1.))
    GRU.bias.data = torch.nn.Parameter(torch.zeros(1, HIDDEN_STATE_SIZE))
    expected_hidden_state = torch.tanh(x.sum()).expand(HIDDEN_STATE_SIZE)

    # Run the forward pass to get the hidden state
    new_hidden_state = GRU(x, hidden_state.clone()).squeeze()

    assert torch.isclose(expected_hidden_state, new_hidden_state).all(), \
        'Update gate test failed when update gate was CLOSED via BIAS and new hidden state weights were 1 with bias 0.'

    # Turn off the update gate (via weights) so that the output is the provided input hidden state
    GRU.bias_u.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    GRU.weight_u.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(-1000.))

    # Run the forward pass to get the hidden state
    new_hidden_state = GRU(x, hidden_state.clone()).squeeze()

    assert torch.isclose(expected_hidden_state, new_hidden_state).all(), \
        'Update gate test failed with update gate CLOSED via WEIGHTS, new hidden state weights 1 and bias 0.'

    # Turn on the update gate only partially by setting both weights and biases to 0 as sigmoid(0) = 0.5
    GRU.bias_u.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    GRU.weight_u.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))

    # Run the forward pass to get the hidden state
    new_hidden_state = GRU(x, hidden_state.clone()).squeeze()
    # The expected state now is should be half of the input one and half of the new updated one
    expected_hidden_state = (0.5 * expected_hidden_state) + (0.5 * hidden_state.squeeze())

    assert torch.isclose(expected_hidden_state, new_hidden_state).all(), \
        'Update gate test failed with update gate PARTIALLY OPEN, new hidden state weights 1 and bias 0.'


# These values were collected for the test below by running the test on the correct implementation of the GRU.
# The 729 different combinations of all weights and biases should cover most basic cases.
# I kept the values of x, hidden_state fixed to limit the number of already very huge test cases
expected_values = [
    -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
    -1., -1., -1., -1., -1., -1., -1., -1., -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902,
    -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902,
    -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902,
    -0.97902, -0.29, 0.39902, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, -1., -1., -1., -1., -1.,
    -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
    -1., -1., -1., -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29,
    0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29,
    0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29,
    0.39902, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, -1., -1., -1., -1., -1., -1., -1., -1., -1.,
    -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
    -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902,
    -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902,
    -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902, -0.97902, -0.29, 0.39902,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, -0.85106, -0.85106, -0.85106, -0.85106, -0.85106,
    -0.85106, -0.85106, -0.85106, -0.85106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85106, 0.85106,
    0.85106, 0.85106, 0.85106, 0.85106, 0.85106, 0.85106, 0.85106, -0.83228, -0.21553, 0.40122,
    -0.83228, -0.21553, 0.40122, -0.83228, -0.21553, 0.40122, 0.0062, 0.21, 0.41379,
    0.0062, 0.21, 0.41379, 0.0062, 0.21, 0.41379, 0.8446955, 0.635532, 0.42636, 0.8446955, 0.635532,
    0.42636, 0.8446955, 0.635532, 0.42636, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, -0.862605, -0.862605,
    -0.862605, -0.9915289, -0.9915289, -0.9915289, -0.9995096, -0.9995096, -0.9995096, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.862605, 0.862605, 0.862605, 0.9915289, 0.9915289, 0.9915289, 0.9995096, 0.9995096, 0.9995096,
    -0.84365577, -0.2213025, 0.40105075, -0.970675, -0.28576446, 0.39914602, -0.97853774, -0.2897548, 0.39902812,
    0.0062, 0.21, 0.41379, 0.0062, 0.21, 0.41379, 0.0062, 0.21, 0.41379, 0.8560659, 0.64130247,
    0.42653906, 0.98308516, 0.7057645, 0.4284438, 0.9909479, 0.70975477, 0.4285617, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, -0.99955, -0.99955, -0.99955, -0.99955, -0.99955, -0.99955, -0.99955, -0.99955,
    -0.99955, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99955, 0.99955, 0.99955, 0.99955,
    0.99955, 0.99955, 0.99955, 0.99955, 0.99955, -0.97857, -0.2897752, 0.39902, -0.97857,
    -0.2897752, 0.39902, -0.97857, -0.2897752, 0.39902, 0.0062, 0.21, 0.41379, 0.0062, 0.21,
    0.41379, 0.0062, 0.21, 0.41379, 0.990988, 0.70977515, 0.42856228, 0.990988, 0.70977515, 0.42856228,
    0.990988, 0.70977515, 0.42856228, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.99143,
    0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856,
    0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71,
    0.42856, 0.99143, 0.71, 0.42856, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71,
    0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143,
    0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71,
    0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.99143,
    0.71, 0.42856, 0.99143, 0.71, 0.42856, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42
]
expected_value_generator = (i for i in expected_values)


@pytest.mark.parametrize('weight_u', [-1., 0., 1.])
@pytest.mark.parametrize('weight_r', [-1., 0., 1.])
@pytest.mark.parametrize('weight', [-1., 0., 1.])
@pytest.mark.parametrize('bias_u', [-100., 0., 100.])
@pytest.mark.parametrize('bias_r', [-100., 0., 100.])
@pytest.mark.parametrize('bias', [-100., 0., 100.])
@pytest.mark.parametrize('x', [[i] * INPUT_SIZE for i in [0.42]])
@pytest.mark.parametrize('hidden_state', [[i] * HIDDEN_STATE_SIZE for i in [0.42]])
def test_whole_gru_with_weight_bias_combinations(weight_u, weight_r, weight, bias_u, bias_r, bias, x, hidden_state):
    """
    This test tests various different combinations of all weights and bias values. The weights are either -1, 0 or 1.
    With different weights being 0, certain gates would be half on. Biases being -100 or 100 will keep the gate on
    or off because w * x would be much smaller than 100 to have an effect.
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Fill up the weight values
    GRU.weight_u = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(weight_u))
    GRU.weight_r = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(weight_r))
    GRU.weight = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(weight))
    # Fill up the bias values
    GRU.bias_u = torch.nn.Parameter(bias_u * torch.ones(1, HIDDEN_STATE_SIZE))
    GRU.bias_r = torch.nn.Parameter(bias_r * torch.ones(1, HIDDEN_STATE_SIZE))
    GRU.bias = torch.nn.Parameter(bias * torch.ones(1, HIDDEN_STATE_SIZE))

    # Run the forward pass
    new_hidden_state = GRU(x, hidden_state).squeeze()
    # Create a whole tensor for the expected value
    expected_hidden_state = torch.tensor(next(expected_value_generator)).expand(HIDDEN_STATE_SIZE)

    assert torch.isclose(expected_hidden_state, new_hidden_state, atol=1e-4).all()


@pytest.mark.parametrize('hidden_state_size, input_size', [(7, 7), (5, 10), (10, 5)])
def test_gru_initialisation_shapes(hidden_state_size, input_size):
    gru = GRUCell(hidden_state_size, input_size)
    for w in (gru.weight_r, gru.weight_u, gru.weight):
        assert list((hidden_state_size + input_size, hidden_state_size)) == list(w.shape)
    for b in (gru.bias_u, gru.bias_r, gru.bias):
        assert [1, hidden_state_size] == list(b.shape)
