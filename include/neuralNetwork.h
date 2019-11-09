#define MAX_INPUT_LAYER_SIZE 20
#define MAX_HIDDEN_LAYER_SIZE 40
#define MAX_OUTPUT_LAYER_SIZE 20

// Mainly for easy testing.
#define DEFAULT_IN_NEURONS 5
#define DEFAULT_HIDDEN_NEURONS 5
#define DEFAULT_OUT_NEURONS 5
#define DEFAULT_EPSILON 1.0
#define DEFAULT_LEARNINGRATE 0.5

/**
 * Neural network with backpropagation.
 */
class neuralNetwork {
   private:
    // Network structure:
    const int inNeurons = 0;
    const int hiddenNeurons = 0;
    const int outNeurons = 0;

    // Learning
    double epsilon = 1.0;
    double learningrate = 0.5;

    // Arbitraly choosen restrictions...
    double inputLayer[MAX_INPUT_LAYER_SIZE + 1];
    double hiddenLayer[MAX_HIDDEN_LAYER_SIZE + 1];
    double outputLayer[MAX_OUTPUT_LAYER_SIZE + 1];

   public:
    /**
     * Constructors
     */
    neuralNetwork()
        : inNeurons(DEFAULT_IN_NEURONS),
          hiddenNeurons(DEFAULT_HIDDEN_NEURONS),
          outNeurons(DEFAULT_OUT_NEURONS),
          epsilon(DEFAULT_EPSILON),
          learningrate(DEFAULT_LEARNINGRATE){};
    neuralNetwork(int inNeurons, int hiddenNeurons, int outNeurons)
        : inNeurons(inNeurons),
          hiddenNeurons(hiddenNeurons),
          outNeurons(outNeurons),
          epsilon(DEFAULT_EPSILON),
          learningrate(DEFAULT_LEARNINGRATE){};
    neuralNetwork(int inNeurons, int hiddenNeurons, int outNeurons,
                  double epsilon, double learningrate)
        : inNeurons(inNeurons),
          hiddenNeurons(hiddenNeurons),
          outNeurons(outNeurons),
          epsilon(epsilon),
          learningrate(learningrate){};
    /**
     * Getters
     */
    int getInNeurons() { return inNeurons; };
    int getHiddenNeurons() { return hiddenNeurons; };
    int getOutNeurons() { return outNeurons; };
    double getEpsilon() { return epsilon; };
    double getLearningrate() { return learningrate; };
    /**
     * Various activation functions
     */
    double calcSigmoid(int x);
    double calcFastSigmoid(int x);
    double calcRelu(int x);
    double calcRelu6(int x);
    /**
     * Destructor
     */
    ~neuralNetwork(){};
};
