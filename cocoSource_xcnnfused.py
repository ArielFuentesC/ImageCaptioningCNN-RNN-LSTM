import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class ImageCaptionModel(nn.Module):
    def __init__(self, config: dict):
        """
        This is the main module class for the image captioning network
        :param config: dictionary holding neural network configuration
        """
        super(ImageCaptionModel, self).__init__()
        # Store config values as instance variables
        self.vocabulary_size = config['vocabulary_size']
        self.embedding_size = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers = config['num_rnn_layers']
        self.cell_type = config['cellType']

        # Create the network layers
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_size)
        # TODO: The output layer (final layer) is a linear layer. What should be the size (dimensions) of its output?
        #         Replace None with a linear layer with correct output size
        self.output_layer = nn.Linear(self.hidden_state_sizes, self.vocabulary_size)  # nn.Linear(self.hidden_state_sizes, )
        self.nn_map_size = 512  # The output size for the image features after the processing via self.inputLayer
        # TODO: Check the task description and replace None with the correct input layer
        self.input_layer = nn.Sequential(nn.Dropout(p=0.25),
                                         nn.Linear(self.number_of_cnn_features, self.nn_map_size),
                                         nn.LeakyReLU())

        self.simplified_rnn = True
        
        if self.simplified_rnn:
            # Simplified one layer RNN is used for task 1 only.
            if self.cell_type != 'RNN':
                raise ValueError('config["cellType"] must be "RNN" when self.simplified_rnn has been set to True.'
                                 'It is ', self.cell_type, 'instead.')

            if self.num_rnn_layers != 1:
                raise ValueError('config["num_rnn_layers"] must be 1 for simplified RNN.'
                                 'It is', self.num_rnn_layers, 'instead.')

            self.rnn = RNNOneLayerSimplified(input_size=self.embedding_size + self.nn_map_size,
                                             hidden_state_size=self.hidden_state_sizes)
        else:
            self.rnn = RNN(input_size=self.embedding_size + self.nn_map_size,
                           hidden_state_size=self.hidden_state_sizes,
                           num_rnn_layers=self.num_rnn_layers,
                           cell_type=self.cell_type)

    def forward(self, cnn_features, x_tokens, is_train: bool, current_hidden_state=None) -> tuple:
        """
        :param cnn_features: Features from the CNN network, shape[batch_size, number_of_cnn_features]
        :param x_tokens: Shape[batch_size, truncated_backprop_length]
        :param is_train: A flag used to select whether or not to use estimated token as input
        :param current_hidden_state: If not None, it should be passed into the rnn module. It's shape should be
                                    [num_rnn_layers, batch_size, hidden_state_sizes].
        :return: logits of shape [batch_size, truncated_backprop_length, vocabulary_size] and new current_hidden_state
                of size [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # HINT: For task 4, you might need to do self.input_layer(torch.transpose(cnn_features, 1, 2))
        processed_cnn_features = self.input_layer(cnn_features)

        if current_hidden_state is None:
            # TODO: Initialize initial_hidden_state with correct dimensions depending on the cell type.
            # The shape of the hidden state here should be [num_rnn_layers, batch_size, hidden_state_sizes].
            # Remember that each rnn cell needs its own initial state.
            batch_size = x_tokens.shape[0]
            initial_hidden_state = Variable(torch.zeros([self.num_rnn_layers, batch_size, self.hidden_state_sizes]))
        else:
            initial_hidden_state = current_hidden_state

        # Call self.rnn to get the "logits" and the new hidden state
        logits, hidden_state = self.rnn(x_tokens, processed_cnn_features, initial_hidden_state, self.output_layer,
                                        self.embedding_layer, is_train)

        return logits, hidden_state

######################################################################################################################


class RNNOneLayerSimplified(nn.Module):
    def __init__(self, input_size, hidden_state_size):
        super(RNNOneLayerSimplified, self).__init__()

        self.input_size = input_size
        self.hidden_state_size = hidden_state_size

        self.cells = nn.ModuleList(
            [RNNsimpleCell(hidden_state_size=self.hidden_state_size, input_size=self.input_size)])
        
    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.

        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length
        else:
            sequence_length = 40  # Max sequence length to be generated
            
        # Get embeddings for the whole sequence
        all_embeddings = embedding_layer(input=tokens)  # Should've shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state.to('cuda')
        # TODO: Fetch the first (index 0) embeddings that should go as input to the RNN.
        # Use these tokens in the loop(s) below
        current_time_step_embeddings = all_embeddings[:,0,:]  # Should have shape (batch_size, embedding_size)

        # Use for loops to run over "sequence_length" and "self.num_rnn_layers" to compute logits
        for i in range(sequence_length):
            # This is for a one-layer RNN
            # In a two-layer RNN you need to iterate through the 2 layers
            # The input for the 2nd layer will be the output (hidden state) of the 1st layer
            # TODO: Create the input for the RNN cell
            input_for_the_first_layer = torch.cat((current_time_step_embeddings, processed_cnn_features), dim=1)
            # Note that the current_hidden_state has 3 dims i.e. len(current_hidden_state.shape) == 3
            # with first dimension having only 1 element, while the RNN cell needs a state with 2 dims as input
            # TODO: Call the RNN cell with input_for_the_first_layer and current_hidden_state as inputs
            #       Hint: Unsqueeze the output
            #x: tensor with shape [batch_size, inputSize]
            #state_old: tensor with shape [batch_size, hidden_state_sizes]
            current_hidden_state = self.cells[0](input_for_the_first_layer, current_hidden_state[0, :, :]).unsqueeze(0)

            # For a multi-layer RNN, apply the output layer (as done below) only after the last layer of the RNN
            # NOTE: for LSTM you use only the part(1st half of the tensor) which corresponds to the hidden state
            logits_i = output_layer(current_hidden_state[0, :, :])
            logits_sequence.append(logits_i)
            # Find the next predicted output element
            predictions = torch.argmax(logits_i, dim=1)

            # Set the embeddings for the next time step
            # training:  the next vector from embeddings which comes from the input sequence
            # prediction/inference: the last predicted token
            if i < sequence_length - 1:
                if is_train:
                    current_time_step_embeddings = all_embeddings[:, i+1, :]
                else:
                    current_time_step_embeddings = embedding_layer(predictions)

        logits = torch.stack(logits_sequence, dim=1)  # Convert the sequence of logits to a tensor

        return logits, current_hidden_state


class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='GRU'):
        """
        :param input_size: Size of the embeddings
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        # TODO: len(input_size_list) == num_rnn_layers and input_size_list[i] should contain the input size for layer i.
        # This is used to populate self.cells
        input_size_list = [self.input_size]
        for i in range(self.num_rnn_layers- 1):
            input_size_list.append(self.hidden_state_size)

        # TODO: Create a list of type "nn.ModuleList" and populate it with cells of type
        #       "self.cell_type" - depending on the number of RNN layers.
        if self.cell_type == "GRU":
            self.cells = nn.ModuleList(
                [GRUCell(hidden_state_size=self.hidden_state_size, input_size=i) for i in input_size_list])

        if self.cell_type == "LSTM":
            self.cells = nn.ModuleList(
                [LSTMCell(hidden_state_size=self.hidden_state_size, input_size=i) for i in input_size_list])

    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.

        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length
        else:
            sequence_length = 40  # Max sequence length to generate

        # Get embeddings for the whole sequence
        embeddings = embedding_layer(input=tokens)  # Should have shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state
        # TODO: Fetch the first (index 0) embeddings that should go as input to the RNN.
        # Use these tokens in the loop(s) below
        input_tokens = embeddings[:,0,:]  # Should have shape (batch_size, embedding_size)
        for i in range(sequence_length):
            hidden_state = Variable(torch.zeros_like(current_hidden_state))
            # TODO:
            # 1. Loop over the RNN layers and provide them with correct input. Inputs depend on the layer
            #    index so input for layer-0 will be different from the input for other layers.
            input_first_layer = torch.cat((input_tokens, processed_cnn_features), dim=1)
            # 2. Update the hidden cell state for every layer.
            current_hidden_state[0, :] = self.cells(input_first_layer, hidden_state)
            # 3. If you are at the last layer, then produce logits_i, predictions. Append logits_i to logits_sequence.
            #    See the simplified rnn for the one layer version.
            logits_i = output_layer(hidden_state[0, :])
            logits_sequence.append(logits_i)
            # Get the input tokens for the next step in the sequence
            predictions = torch.argmax(logits_i, dim=1)
            if i < sequence_length - 1:
                if is_train:
                    input_tokens = embeddings[:, i+1, :]
                else:
                    # TODO: Compute predictions above and use them here by replacing None with the code in comment
                    input_tokens = embedding_layer(predictions)

        logits = torch.stack(logits_sequence, dim=1)  # Convert the sequence of logits to a tensor

        return logits, current_hidden_state

########################################################################################################################


class GRUCell(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to the GRU
        """
        super(GRUCell, self).__init__()
        self.hidden_state_sizes = hidden_state_size

        # TODO: Initialise weights and biases for the update gate (weight_u, bias_u), reset gate (w_r, b_r) and hidden
        #       state (weight, bias).
        #       self.weight, self.weight_(u, r):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(u, r): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n

        # Initialize weights and biases for the update gate (weight_u, bias_u), reset gate (w_r, b_r) and hidden
        # state (weight, bias) using variance scaling initialization.
        n = hidden_state_size + input_size

        # Update gate parameters
        self.weight_u = nn.Parameter(torch.empty(n, hidden_state_size))
        nn.init.xavier_normal_(self.weight_u)
        self.bias_u = nn.Parameter(torch.zeros(1, hidden_state_size))

        # Reset gate parameters
        self.weight_r = nn.Parameter(torch.empty(n, hidden_state_size))
        nn.init.xavier_normal_(self.weight_r)
        self.bias_r = nn.Parameter(torch.zeros(1, hidden_state_size))

        # Hidden state parameters
        self.weight = nn.Parameter(torch.empty(n, hidden_state_size))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, hidden_state_size))

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for a GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, HIDDEN_STATE_SIZE]
        :return: The updated hidden state of the GRU cell. Shape: [batch_size, HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the GRU equations to get the new hidden state and return it
        # Compute update gate
        update_gate = torch.sigmoid(torch.matmul(torch.cat([x, hidden_state], dim=1), self.weight_u) + self.bias_u)

        # Compute reset gate
        reset_gate = torch.sigmoid(torch.matmul(torch.cat([x, hidden_state], dim=1), self.weight_r) + self.bias_r)

        # Compute hidden state candidate
        hidden_state_candidate = torch.tanh(
            torch.matmul(torch.cat([x, reset_gate * hidden_state], dim=1), self.weight) + self.bias)

        # Compute new hidden state
        new_hidden_state = update_gate * hidden_state + (1 - update_gate) * hidden_state_candidate

        return new_hidden_state

######################################################################################################################


class RNNsimpleCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            input_size: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes + input_size, hidden_state_sizes]. Initialized
                         using variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        super(RNNsimpleCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        self.weight = nn.Parameter(
            torch.randn(input_size + hidden_state_size, hidden_state_size) / np.sqrt(input_size + hidden_state_size))
        self.bias = nn.Parameter(torch.zeros(1, hidden_state_size))

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        x2 = torch.cat((x, state_old), dim=1)
        state_new = torch.tanh(torch.mm(x2, self.weight) + self.bias)
        return state_new

######################################################################################################################


class LSTMCell(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to GRU
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        # TODO: Initialise weights and biases for the forget gate (weight_f, bias_f), input gate (w_i, b_i),
        #       output gate (w_o, b_o), and hidden state (weight, bias)
        #       self.weight, self.weight_(f, i, o):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(f, i, o): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to two.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n
        #       Note: The actual input tensor will have 2 * HIDDEN_STATE_SIZE because it contains both
        #             hidden state and cell's memory

        # Forget gate parameters
        self.weight_f = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(2 * hidden_state_size + input_size, 2 * hidden_state_size)))
        self.bias_f = nn.Parameter(torch.zeros((1, hidden_state_size))) + 2

        # Input gate parameters
        self.weight_i = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(2 * hidden_state_size + input_size, 2 * hidden_state_size)))
        self.bias_i = nn.Parameter(torch.zeros((1, 2 * hidden_state_size))) + 2

        # Output gate parameters
        self.weight_o = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(2 * hidden_state_size + input_size, 2 * hidden_state_size)))
        self.bias_o = nn.Parameter(torch.zeros((1, 2 * hidden_state_size))) + 2

        # Memory cell parameters
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(2 * hidden_state_size + input_size, 2 * hidden_state_size)))
        self.bias = nn.Parameter(torch.zeros((1, 2 * hidden_state_size))) + 2

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for an GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the GRU cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the GRU equations to get the new hidden state, cell memory and return them.
        #       The first half of the returned value must represent the new hidden state and the second half
        #       new cell state.
        # Split hidden state into the previous hidden state and cell memory
        prev_hidden_state, prev_cell_memory = hidden_state[:, :self.hidden_state_size], \
            hidden_state[:, self.hidden_state_size:]

        # Concatenate the input and the previous hidden state
        combined_input = torch.cat([x, prev_hidden_state], dim=1)

        # Compute the forget gate
        forget_gate = torch.sigmoid(torch.matmul(combined_input, self.weight_f) + self.bias_f)
        # Compute the input gate
        input_gate = torch.sigmoid(torch.matmul(combined_input, self.weight_i) + self.bias_i)
        # Compute the candidate cell memory
        candidate_memory = torch.tanh(torch.matmul(combined_input, self.weight) + self.bias)
        # Compute the updated cell memory
        cell_memory = forget_gate * prev_cell_memory + input_gate * candidate_memory
        # Compute the output gate
        output_gate = torch.sigmoid(torch.matmul(combined_input, self.weight_o) + self.bias_o)
        # Compute the updated hidden state
        new_hidden_state = output_gate * torch.tanh(cell_memory)

        # Concatenate the new hidden state and cell memory to return the updated hidden state (including memory)
        return torch.cat([new_hidden_state, cell_memory], dim=1)
        

######################################################################################################################

def loss_fn(logits, y_tokens, y_weights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits           : Shape[batch_size, truncated_backprop_length, vocabulary_size]
        y_tokens (labels): Shape[batch_size, truncated_backprop_length]
        y_weights         : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only
                           from words existing
                           (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sum_loss: The total cross entropy loss for all words
        mean_loss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 1e-7  # Used to avoid division by 0
    #print(logits.shape, y_tokens.shape)
    #logits = logits[0, :, :, :]
    logits = logits.view(-1, logits.shape[2])
    y_tokens = y_tokens.reshape(-1) #y_tokens.view(-1)
    y_weights = y_weights.reshape(-1) #y_weights.view(-1)
    #print(y_tokens.shape)
    #print(y_weights.shape)
    #print(logits.shape, y_tokens.shape)
    losses = F.cross_entropy(input=logits, target=y_tokens, reduction='none')

    sum_loss = (losses * y_weights).sum()
    mean_loss = sum_loss / (y_weights.sum() + eps)

    return sum_loss, mean_loss


# #####################################################################################################################
#if __name__ == '__main__':
#
#     lossDict = {'logits': logits,
#                 'yTokens': yTokens,
#                 'yWeights': yWeights,
#                 'sumLoss': sumLoss,
#                 'meanLoss': meanLoss
#     }
#
#     sumLoss, meanLoss = loss_fn(logits, yTokens, yWeights)
#



