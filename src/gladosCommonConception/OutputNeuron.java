package gladosCommonConception;

/**
 * An output neuron for a neural network. Uses an inputLayer but no outputLayer,
 * otherwise works exactly like an <code>IntermediateNeuron</code>.
 * 
 * @author Laty
 *
 */
public class OutputNeuron extends ActiveNeuron {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -3600188553007839719L;

	public OutputNeuron(int finalInputSize) {
		super(finalInputSize);
	}
	public OutputNeuron(boolean deterministic){
		super(deterministic);
	}


}
