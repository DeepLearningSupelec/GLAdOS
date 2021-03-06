package gladosCommonConception;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JRadioButton;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JSlider;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

/**
 * This visual window is executable and allows the training of a new
 * <code>NeuralNetwork</code> from scratch and following a number of parameters.
 * The NeuralNetwork always has one and only one hidden layer because this has
 * been found to be the quickest and most reliable type of NeuralNetwork.
 * 
 * @author Laty
 *
 */
public class LearningWindow extends ApplicationFrame {

	private static final long serialVersionUID = 1L;

	private JTextField txtDefaultLearningRate;
	private JTextField txtLearningProportion;
	private JTextField txtTargetErrorPer;
	private JTextField txtIncreaseAndDecrease;
	private JTextField txtMomentumFactor;

	private XYSeries errorSeries = new XYSeries("Learning");
	private XYSeries mistakeSeries = new XYSeries("Learning");
	private XYSeries testErrorSeries = new XYSeries("Test");
	private XYSeries testMistakeSeries = new XYSeries("Test");

	private ChartPanel graphPanel;
	private ChartPanel mistakePanel;
	private JTextField txtEpochSize;
	private JTextField txtRefreshRate;
	private JTextField txtNumberOfSamples;
	private JTextField txtNumberOfAlgorithm;
	private JTextField iterationTxt;

	public static void main(String[] args) {
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (UnsupportedLookAndFeelException e) {
			e.printStackTrace();
		}
		new LearningWindow("Test");
	}

	public void massLearningAlg(boolean simple, boolean variableLR, boolean preprocessing, boolean incrementPerEpoch,
			double defaultLR, double decreaseLR_factor, double increaseLR_factor, double momentumRate,
			double learningProportion, int targetSampleNumber, int epochSize, int refreshSize, int iterationNumber) {

		Path trainImages = FileSystems.getDefault().getPath("src/filesMNIST", "train-images.idx3-ubyte");
		Path trainLabels = FileSystems.getDefault().getPath("src/filesMNIST", "train-labels.idx1-ubyte");
		Preprocessing processedFile = null;
		if (preprocessing) {
			processedFile = new Preprocessing();
		}
		byte[] rawImagesArray = null;
		byte[] labelsArray = null;
		if (!preprocessing) {
			try {
				rawImagesArray = Files.readAllBytes(trainImages);
				labelsArray = Files.readAllBytes(trainLabels);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		@SuppressWarnings("unused")
		double averageErrorPerEpoch = Double.MAX_VALUE;
		// double lastAverageEpochError = Double.MAX_VALUE;
		double averageTestErrorPerEpoch = Double.MAX_VALUE;
		double learningRate = defaultLR;
		double[] expectedOutput = null;
		double[] input;

		int[] currentPermutation;
		SourceImage currentImage;
		int sampleNumber = 0;
		Output out = new Output();

		if (!simple) {
			// int i = 0;
			// int numberOfMistakesPerEpoch = 0;
			// int numberOfTestMistakesPerEpoch = 0;
			for (int k = 0; k < iterationNumber; k++) {
				iterationTxt.setText((k + 1) + " / " + iterationNumber);
				FeedForward learningNN;
				List<SourceImage> cleanInput;
				if (preprocessing) {
					learningNN = new FeedForward(new int[] { 100, 50, 10 }, learningRate);
					cleanInput = recreateCleanInput(processedFile.getFeatures(), processedFile.getExpectedOutputs());
				} else {
					learningNN = new FeedForward(new int[] { 28 * 28, 100, 40, 10 }, learningRate);
					cleanInput = createCleanInput(rawImagesArray, labelsArray);
				}
				cleanInput = permutateRandomly(cleanInput);
				int refreshCounter = 0;
				int epochCounter = 0;
				int rawNumber = 0;
				sampleNumber = 0;
				int learningSize = (int) (cleanInput.size() * learningProportion);
				currentPermutation = getRandomPermutation(learningSize);
				
				while (sampleNumber < targetSampleNumber) {
					
					currentImage = cleanInput.get(currentPermutation[rawNumber]);

					if (preprocessing) {
						input = currentImage.getRelevantFeatures();
					} else {
						input = currentImage.getCleanRawDoubleImage();
					}
					learnOneInput(learningNN, input, incrementPerEpoch, momentumRate, learningRate, currentImage,
							variableLR);

					if (refreshCounter == refreshSize - 1
					 ||(sampleNumber == 0 && rawNumber == 0)
					) {
						averageTestErrorPerEpoch = testLearningNN(cleanInput.size(), learningSize, cleanInput,
								learningNN, preprocessing, sampleNumber, refreshSize, out, currentPermutation);
						sampleNumber++;
						update(getGraphics());
						refreshCounter = 0;
					}

					if (epochCounter == epochSize - 1) {
						if (!incrementPerEpoch) {

						}
						currentPermutation = getRandomPermutation(learningSize);
						rawNumber = 0;
					}
					if (rawNumber == learningSize - 1) {
						rawNumber = 0;
						currentPermutation = getRandomPermutation(learningSize);
					}
					refreshCounter++;
					epochCounter++;
					rawNumber++;

					
				}
				mistakeSeries.clear();
				errorSeries.clear();
				testMistakeSeries.clear();
				testErrorSeries.clear();

				update(getGraphics());
				revalidate();
			}
			JFileChooser chooser = new JFileChooser(".");

			int returnVal = chooser.showSaveDialog(this);

			try {
				if (returnVal == JFileChooser.APPROVE_OPTION && chooser.getSelectedFile() != null) {
					// FileOutputStream fileOut;
					if (preprocessing) {
						if (chooser.getSelectedFile().getPath().endsWith(".pre")) {
							// fileOut = new
							// FileOutputStream(chooser.getSelectedFile().getPath());
							out.generateCsvFile((chooser.getSelectedFile().getPath()).substring(0,
									(chooser.getSelectedFile().getPath()).length() - 4));
						} else {
							// fileOut = new
							// FileOutputStream(chooser.getSelectedFile().getPath()
							// + ".pre");
							out.generateCsvFile(chooser.getSelectedFile().getPath());
						}

					} else {
						if (chooser.getSelectedFile().getPath().endsWith(".rw")) {
							// fileOut = new FileOutputStream("resultingNN/" +
							// chooser.getSelectedFile().getPath());
						} else {
							// fileOut = new FileOutputStream(
							// "resultingNN/" +
							// chooser.getSelectedFile().getPath() + ".rw");
						}
					}
					// ObjectOutputStream outStream = new
					// ObjectOutputStream(fileOut);
					// outStream.writeObject(learningNN);
					// outStream.close();
					// fileOut.close();
				}

			} catch (Exception e) {

				e.printStackTrace();
			}
			this.dispose();
		}

		else {
			FeedForward learningNN = new FeedForward(new int[] { 2, 3, 4, 1 }, learningRate);
			for (int i = 0; i < 10000 / 100; i++) {
				for (int j = 0; j < 100; j++) {
					double inputA = Math.floor(Math.random() * 2);
					double inputB = Math.floor(Math.random() * 2);
					learningNN.setInputs(new double[] { inputA, inputB });
					learningNN.fire();
					if (inputA == inputB) {
						expectedOutput = new double[] { inputA };
					} else {
						expectedOutput = new double[] { inputA + inputB };
					}
					backpropagateNeuronDiff(learningNN, expectedOutput);
					// learningNN.calculateNeuronDiffs(expectedOutput);
					// learningNN.incrementWeightDiffs();
					backpropagateWeightDiffs(learningNN);
					// learningNN.incrementWeights();
					backpropagateWeight(learningNN, learningRate);
					// learningNN.resetWeightDiffsMomentum(momentumRate);
					averageErrorPerEpoch += currentError(expectedOutput, learningNN.getOutputs());
				}
				sampleNumber++;

				// averageTestErrorPerEpoch = testLearningNN(cleanInput.size(),
				// learningSize, cleanInput, learningNN,
				// preprocessing, epochNumber);
				update(getGraphics());

				revalidate();
				// learningNN.incrementWeights();
				// //learningNN.resetWeightDiffs();
				// learningNN.resetWeightDiffsMomentum(MOMENTUM_RATE);

				averageErrorPerEpoch = 0.;

			}
		}
	}

	/**
	 * Creates a new Neural network and launches the learning algorithm
	 * following the backpropagation method.
	 * 
	 * @param simple
	 * @param variableLR
	 *            : if true the learning rate will be adaptable, following the
	 *            decreaseLR_factor and increaseLR_factor.
	 * @param preprocessing
	 *            : if true, uses preprocessed images (following the method
	 *            described in the Leon and Sandu article method).
	 * @param incrementPerEpoch
	 *            : if true, increments only every epoch rather than after every
	 *            example.
	 * @param defaultLR
	 *            : the default learning rate.
	 * @param decreaseLR_factor
	 * @param increaseLR_factor
	 * @param momentumRate
	 *            : the momentum factor, if 0, is equivalent to no momentum.
	 * @param epochSize
	 *            : the size of one epoch.
	 * @param targetAverageError
	 *            : the target average error for one epoch (in percentage) after
	 *            which the learning algorithm will stop.
	 */
	public void learningAlg(boolean simple, boolean variableLR, boolean preprocessing, boolean incrementPerEpoch,
			double defaultLR, double decreaseLR_factor, double increaseLR_factor, double momentumRate,
			double learningProportion, double targetAverageError, int epochSize, int refreshSize,
			boolean deterministic) {

		Path trainImages = FileSystems.getDefault().getPath("src/filesMNIST", "train-images.idx3-ubyte");
		Path trainLabels = FileSystems.getDefault().getPath("src/filesMNIST", "train-labels.idx1-ubyte");
		Preprocessing processedFile = null;
		if (preprocessing) {
			processedFile = new Preprocessing();
		}
		byte[] rawImagesArray = null;
		byte[] labelsArray = null;
		if (!preprocessing) {
			try {
				rawImagesArray = Files.readAllBytes(trainImages);
				labelsArray = Files.readAllBytes(trainLabels);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		@SuppressWarnings("unused")
		double averageErrorPerEpoch = Double.MAX_VALUE;
		// double lastAverageEpochError = Double.MAX_VALUE;
		double averageTestErrorPerEpoch = Double.MAX_VALUE;
		double learningRate = defaultLR;
		double[] expectedOutput = null;
		double[] input;

		int[] currentPermutation;
		SourceImage currentImage;
		int sampleNumber = 0;
		Output out = new Output();

		if (!simple) {
			// int i = 0;
			// int numberOfMistakesPerEpoch = 0;
			// int numberOfTestMistakesPerEpoch = 0;
			FeedForward learningNN;
			List<SourceImage> cleanInput;
			if (preprocessing) {
				if (!deterministic) {
					learningNN = new FeedForward(new int[] { 100, 50, 10 }, learningRate);
				} else {
					learningNN = new FeedForward(new int[] { 100, 50, 10 }, learningRate, true);
				}

				cleanInput = recreateCleanInput(processedFile.getFeatures(), processedFile.getExpectedOutputs());
			} else {
				learningNN = new FeedForward(new int[] { 28 * 28, 90, 10 }, learningRate);
				cleanInput = createCleanInput(rawImagesArray, labelsArray);
			}
			if (!deterministic) {
				cleanInput = permutateRandomly(cleanInput);
			}
			int refreshCounter = 0;
			int epochCounter = 0;
			int rawNumber = 0;
			int learningSize = (int) (cleanInput.size() * learningProportion);
			if (!deterministic) {
				currentPermutation = getRandomPermutation(learningSize);
			} else {
				currentPermutation = identity(learningSize);
			}
			
			while (sampleNumber < 50) {
				currentImage = cleanInput.get(currentPermutation[rawNumber]);

				if (preprocessing) {
					input = currentImage.getRelevantFeatures();
				} else {
					input = currentImage.getCleanRawDoubleImage();
				}
				learnOneInput(learningNN, input, incrementPerEpoch, momentumRate, learningRate, currentImage,
						variableLR);

				if (refreshCounter == refreshSize - 1
				 ||(sampleNumber == 0 && rawNumber == 0)
				) {
					averageTestErrorPerEpoch = testLearningNN(cleanInput.size(), learningSize, cleanInput, learningNN,
							preprocessing, sampleNumber, refreshSize, out, currentPermutation);
					sampleNumber++;
					update(getGraphics());
					refreshCounter = 0;
				}

				if (epochCounter == epochSize - 1) {
					if (!incrementPerEpoch) {

					}
					if(!deterministic){
						currentPermutation = getRandomPermutation(learningSize);
					}
					rawNumber = 0;
				}
				if (rawNumber == learningSize - 1) {
					rawNumber = 0;
					if(!deterministic){
						currentPermutation = getRandomPermutation(learningSize);
					}
				}
				refreshCounter++;
				epochCounter++;
				rawNumber++;

				
			}

			update(getGraphics());
			revalidate();

			JFileChooser chooser = new JFileChooser(".");

			int returnVal = chooser.showSaveDialog(this);

			try {
				if (returnVal == JFileChooser.APPROVE_OPTION && chooser.getSelectedFile() != null) {
					FileOutputStream fileOut;
					if (preprocessing) {
						if (chooser.getSelectedFile().getPath().endsWith(".pre")) {
							fileOut = new FileOutputStream(chooser.getSelectedFile().getPath());
							out.generateCsvFile((chooser.getSelectedFile().getPath()).substring(0,
									(chooser.getSelectedFile().getPath()).length() - 4));
						} else {
							fileOut = new FileOutputStream(chooser.getSelectedFile().getPath() + ".pre");
							out.generateCsvFile(chooser.getSelectedFile().getPath());
						}

					} else {
						if (chooser.getSelectedFile().getPath().endsWith(".rw")) {
							fileOut = new FileOutputStream(chooser.getSelectedFile().getPath());
							out.generateCsvFile((chooser.getSelectedFile().getPath()).substring(0,
									(chooser.getSelectedFile().getPath()).length() - 4));
						} else {
							fileOut = new FileOutputStream(chooser.getSelectedFile().getPath() + ".rw");
							out.generateCsvFile(chooser.getSelectedFile().getPath());
						}
					}
					ObjectOutputStream outStream = new ObjectOutputStream(fileOut);
					outStream.writeObject(learningNN);
					outStream.close();
					fileOut.close();
				}

			} catch (Exception e) {

				e.printStackTrace();
			}
			this.dispose();
		}

		else {
			FeedForward learningNN = new FeedForward(new int[] { 2, 3, 4, 1 }, learningRate);
			for (int i = 0; i < 10000 / 100; i++) {
				for (int j = 0; j < 100; j++) {
					double inputA = Math.floor(Math.random() * 2);
					double inputB = Math.floor(Math.random() * 2);
					learningNN.setInputs(new double[] { inputA, inputB });
					learningNN.fire();
					if (inputA == inputB) {
						expectedOutput = new double[] { inputA };
					} else {
						expectedOutput = new double[] { inputA + inputB };
					}
					backpropagateNeuronDiff(learningNN, expectedOutput);
					// learningNN.calculateNeuronDiffs(expectedOutput);
					// learningNN.incrementWeightDiffs();
					backpropagateWeightDiffs(learningNN);
					// learningNN.incrementWeights();
					backpropagateWeight(learningNN, learningRate);
					// learningNN.resetWeightDiffsMomentum(momentumRate);
					averageErrorPerEpoch += currentError(expectedOutput, learningNN.getOutputs());
				}
				sampleNumber++;

				// averageTestErrorPerEpoch = testLearningNN(cleanInput.size(),
				// learningSize, cleanInput, learningNN,
				// preprocessing, epochNumber);
				update(getGraphics());

				revalidate();
				// learningNN.incrementWeights();
				// //learningNN.resetWeightDiffs();
				// learningNN.resetWeightDiffsMomentum(MOMENTUM_RATE);

				averageErrorPerEpoch = 0.;

			}
		}
	}

	private static int[] identity(int learningSize) {
		int[] res = new int[learningSize];
		for (int i = 0; i < learningSize; i++) {
			res[i] = i;
		}
		return res;
	}

	private void learnOneInput(FeedForward learningNN, double[] input, boolean incrementPerEpoch, double momentumRate,
			double learningRate, SourceImage currentImage, boolean variableLR) {
		learningNN.setInputs(input);
		learningNN.fire();
		double[] expectedOutput = currentImage.getExpectedOutput();
		backpropagateNeuronDiff(learningNN, expectedOutput);
		backpropagateWeightDiffs(learningNN);
		if (!incrementPerEpoch) {
			backpropagateWeight(learningNN, learningRate);
			resestWeightDiffs(learningNN, momentumRate);

			if (variableLR) {
				// learningNN.varyLR(decreaseLR_factor,
				// increaseLR_factor);
			}
		}

	}

	private void backpropagateNeuronDiff(FeedForward learningNN, double[] expectedOutput) {
		int c = 0;
		for (AbstractNeuron n : learningNN.getOutputLayer()) {
			n.setNeuronDiff(Sigmoid.getInstance().applyDerivative(((ActiveNeuron) n).getIntermediateValue())
					* (expectedOutput[c] - n.getOutput()));
			c++;
		}
		for (int i = learningNN.getIntermediateLayers().size() - 1; i >= 0; i--) {
			for (AbstractNeuron n : learningNN.getIntermediateLayers().get(i)) {
				double temp = 0;
				for (Synapse s : ((IntermediateNeuron) n).getOutputSynapses()) {
					temp += s.getWeight() * s.getOutputNeuron().getNeuronDiff();
				}
				n.setNeuronDiff(
						Sigmoid.getInstance().applyDerivative(((ActiveNeuron) n).getIntermediateValue()) * temp);
			}
		}
	}

	private void backpropagateWeightDiffs(FeedForward learningNN) {
		for (Synapse s : learningNN.getAllSynapses()) {
			s.setWeightDiff(s.getWeightDiff() + (s.getInputNeuron().getOutput() * s.getOutputNeuron().getNeuronDiff()));
		}
		for (AbstractNeuron n : learningNN.getAllActiveNeurons()) {
			((ActiveNeuron) n).setBiasDiff(((ActiveNeuron) n).getBiasDiff() + n.getNeuronDiff());
		}
	}

	private void backpropagateWeight(FeedForward learningNN, double learningRate) {
		for (Synapse s : learningNN.getAllSynapses()) {
			s.setWeight(s.getWeight() + s.getWeightDiff() * learningRate);
		}
		for (AbstractNeuron n : learningNN.getAllActiveNeurons()) {
			((ActiveNeuron) n).setBias(((ActiveNeuron) n).getBias() + learningRate * ((ActiveNeuron) n).getBiasDiff());
		}
	}

	private void resestWeightDiffs(FeedForward learningNN, double momentumRate) {
		for (Synapse s : learningNN.getAllSynapses()) {
			s.setWeightDiff(s.getWeightDiff() * momentumRate);
		}
		for (AbstractNeuron n : learningNN.getAllActiveNeurons()) {
			((ActiveNeuron) n).setBiasDiff(((ActiveNeuron) n).getBiasDiff() * momentumRate);
		}
	}

	private static double currentError(double[] expectedOutput, List<Double> outputs) {
		double res = 0;
		double temp = 0;
		for (int i = 0; i < expectedOutput.length; i++) {
			temp = Math.abs(expectedOutput[i] - outputs.get(i));
			if (temp > res) {
				res = temp;
			}
		}
		return res;
	}

	/**
	 * Creates the UI window for a learning algorithm.
	 * 
	 * @param title
	 */
	public LearningWindow(final String title) {
		super(title);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setResizable(false);
		setTitle("Visualisation");
		setSize(1000, 529);
		setLocationRelativeTo(null);
		getContentPane().setLayout(null);

		JComboBox<String> simpleBox = new JComboBox<String>();
		simpleBox.setModel((ComboBoxModel<String>) new DefaultComboBoxModel<String>(
				new String[] { "Digit Recognition", "Simple Logic Functon Recognition" }));
		simpleBox.setBounds(10, 22, 193, 20);
		getContentPane().add(simpleBox);

		XYSeriesCollection errorData = new XYSeriesCollection(errorSeries);
		XYSeriesCollection testErrorData = new XYSeriesCollection(testErrorSeries);
		JFreeChart errorChart = ChartFactory.createXYLineChart("Average maximum error per sample (%)", "Sample number",
				"Average maximum error (%)", errorData);
		errorChart.getXYPlot().setDataset(0, errorData);
		errorChart.getXYPlot().setDataset(1, testErrorData);
		XYLineAndShapeRenderer redRenderer = new XYLineAndShapeRenderer();
		XYLineAndShapeRenderer blueRenderer = new XYLineAndShapeRenderer();
		errorChart.getXYPlot().setRenderer(1, blueRenderer);
		errorChart.getXYPlot().setRenderer(0, redRenderer);
		graphPanel = new ChartPanel(errorChart);
		graphPanel.setBounds(10, 189, 464, 279);
		getContentPane().add(graphPanel);

		XYSeriesCollection mistakeData = new XYSeriesCollection(mistakeSeries);
		XYSeriesCollection testMistakeData = new XYSeriesCollection(testMistakeSeries);
		JFreeChart mistakeChart = ChartFactory.createXYLineChart("Mistakes per sample (%)", "Sample number",
				"Mistakes per epoch (%)", mistakeData);
		mistakeChart.getXYPlot().setDataset(0, mistakeData);
		mistakeChart.getXYPlot().setDataset(1, testMistakeData);
		mistakeChart.getXYPlot().setRenderer(0, redRenderer);
		mistakeChart.getXYPlot().setRenderer(1, blueRenderer);
		mistakePanel = new ChartPanel(mistakeChart);

		mistakePanel.setBounds(509, 189, 464, 279);
		getContentPane().add(mistakePanel);

		JRadioButton rdbtnPreprocessing = new JRadioButton("Preprocessing");
		rdbtnPreprocessing.setSelected(true);
		rdbtnPreprocessing.setBounds(10, 75, 93, 23);
		getContentPane().add(rdbtnPreprocessing);

		JRadioButton rdbtnIncrementEveryTime = new JRadioButton("Increment every time");
		rdbtnIncrementEveryTime.setSelected(true);
		rdbtnIncrementEveryTime.setBounds(10, 127, 139, 23);
		getContentPane().add(rdbtnIncrementEveryTime);

		JSpinner spinnerDecreaseLR = new JSpinner();
		spinnerDecreaseLR.setModel(new SpinnerNumberModel(0.8, 0.0, 1.0, 0.1));
		spinnerDecreaseLR.setBounds(368, 102, 46, 20);
		getContentPane().add(spinnerDecreaseLR);

		JSpinner spinnerIncreaseLR = new JSpinner();
		spinnerIncreaseLR.setModel(new SpinnerNumberModel(1.2, new Double(1), null, new Double(0.1)));
		spinnerIncreaseLR.setBounds(427, 102, 47, 20);
		getContentPane().add(spinnerIncreaseLR);

		JRadioButton rdbtnAdaptableLearningRate = new JRadioButton("Adaptable learning rate");
		rdbtnAdaptableLearningRate.setSelected(true);
		rdbtnAdaptableLearningRate.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				txtIncreaseAndDecrease.setEnabled(rdbtnAdaptableLearningRate.isSelected());
				spinnerIncreaseLR.setEnabled(rdbtnAdaptableLearningRate.isSelected());
				spinnerDecreaseLR.setEnabled(rdbtnAdaptableLearningRate.isSelected());

			}
		});
		rdbtnAdaptableLearningRate.setBounds(10, 101, 139, 23);
		getContentPane().add(rdbtnAdaptableLearningRate);

		JSpinner spinnerMomentumFact = new JSpinner();
		spinnerMomentumFact.setModel(new SpinnerNumberModel(0.4, 0.0, 1.0, .1));
		spinnerMomentumFact.setBounds(417, 76, 57, 20);
		getContentPane().add(spinnerMomentumFact);

		JSpinner spinnerLR = new JSpinner();
		spinnerLR.setModel(new SpinnerNumberModel(0.0, null, 1.0, 0.0));
		spinnerLR.setBounds(131, 48, 71, 20);
		getContentPane().add(spinnerLR);

		txtDefaultLearningRate = new JTextField();
		txtDefaultLearningRate.setBorder(null);
		txtDefaultLearningRate.setEditable(false);
		txtDefaultLearningRate.setText("Default learning rate");
		txtDefaultLearningRate.setBounds(20, 48, 101, 20);
		getContentPane().add(txtDefaultLearningRate);
		txtDefaultLearningRate.setColumns(10);

		JSpinner spinnerLearningProp = new JSpinner();
		spinnerLearningProp
				.setModel(new SpinnerNumberModel(new Double(0.8), new Double(0.), new Double(1.), new Double(0.1)));
		spinnerLearningProp.setBounds(428, 48, 46, 20);
		getContentPane().add(spinnerLearningProp);

		JSpinner spinnerEpochSize = new JSpinner();

		JSpinner spinnerRefreshRate = new JSpinner();
		spinnerRefreshRate.setModel(new SpinnerNumberModel(new Integer(5000), new Integer(1), null, new Integer(100)));
		spinnerRefreshRate.setBounds(403, 128, 71, 20);
		getContentPane().add(spinnerRefreshRate);

		JSlider sliderLearningProportion = new JSlider();
		sliderLearningProportion.setMajorTickSpacing(50);
		sliderLearningProportion.setPaintTicks(true);
		sliderLearningProportion.setValue(80);
		sliderLearningProportion.setBounds(342, 48, 85, 19);
		getContentPane().add(sliderLearningProportion);

		spinnerLearningProp.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {

				sliderLearningProportion.setValue((int) ((double) spinnerLearningProp.getValue() * 100));
				double learningProp = ((double) spinnerLearningProp.getValue());
				if ((int) spinnerEpochSize.getValue() > (learningProp * 60000.)) {
					spinnerEpochSize.setValue((int) ((double) spinnerLearningProp.getValue() * 60000.));
				}
				if ((int) spinnerRefreshRate.getValue() > ((double) learningProp * 60000.)) {
					spinnerRefreshRate.setValue((int) (double) spinnerLearningProp.getValue() * 60000.);
				}
				spinnerEpochSize.setModel(new SpinnerNumberModel((int) spinnerEpochSize.getValue(), 1,
						(int) (learningProp * 60000.), 100));
				spinnerRefreshRate.setModel(new SpinnerNumberModel((int) spinnerRefreshRate.getValue(), 1,
						(int) (learningProp * 60000.), 100));
			}
		});

		sliderLearningProportion.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				// System.out.println((int)((double)spinnerLearningProp.getValue()*60000.));

				spinnerLearningProp.setValue(sliderLearningProportion.getValue() / 100.);
				double learningProp = ((double) spinnerLearningProp.getValue());
				if ((int) spinnerEpochSize.getValue() > (learningProp * 60000.)) {
					spinnerEpochSize.setValue((int) ((double) spinnerLearningProp.getValue() * 60000.));
				}
				if ((int) spinnerRefreshRate.getValue() > ((double) learningProp * 60000)) {
					spinnerRefreshRate.setValue((int) ((double) spinnerLearningProp.getValue() * 60000.));
				}
				spinnerEpochSize.setModel(new SpinnerNumberModel((int) spinnerEpochSize.getValue(), 1,
						(int) (learningProp * 60000.), 100));
				spinnerRefreshRate.setModel(new SpinnerNumberModel((int) spinnerRefreshRate.getValue(), 1,
						(int) (learningProp * 60000.), 100));

			}
		});

		txtLearningProportion = new JTextField();
		txtLearningProportion.setBorder(null);
		txtLearningProportion.setHorizontalAlignment(SwingConstants.TRAILING);
		txtLearningProportion.setEditable(false);
		txtLearningProportion.setText("Learning proportion");
		txtLearningProportion.setBounds(212, 48, 120, 20);
		getContentPane().add(txtLearningProportion);
		txtLearningProportion.setColumns(10);

		JSpinner spinnerTarget = new JSpinner();
		spinnerTarget.setModel(new SpinnerNumberModel(new Double(10), new Double(0), new Double(100), new Double(1)));
		spinnerTarget.setBounds(417, 22, 57, 20);
		getContentPane().add(spinnerTarget);

		txtTargetErrorPer = new JTextField();
		txtTargetErrorPer.setBorder(null);
		txtTargetErrorPer.setOpaque(false);
		txtTargetErrorPer.setFocusable(false);
		txtTargetErrorPer.setHorizontalAlignment(SwingConstants.TRAILING);
		txtTargetErrorPer.setEditable(false);
		txtTargetErrorPer.setText("Target average error per epoch (%)");
		txtTargetErrorPer.setBounds(226, 22, 181, 20);
		getContentPane().add(txtTargetErrorPer);
		txtTargetErrorPer.setColumns(10);

		txtIncreaseAndDecrease = new JTextField();
		txtIncreaseAndDecrease.setBorder(null);
		txtIncreaseAndDecrease.setHorizontalAlignment(SwingConstants.TRAILING);
		txtIncreaseAndDecrease.setEditable(false);
		txtIncreaseAndDecrease.setText("Decrease and increase LR factors");
		txtIncreaseAndDecrease.setBounds(188, 102, 170, 20);
		getContentPane().add(txtIncreaseAndDecrease);
		txtIncreaseAndDecrease.setColumns(10);

		txtMomentumFactor = new JTextField();
		txtMomentumFactor.setBorder(null);
		txtMomentumFactor.setEditable(false);
		txtMomentumFactor.setText("Momentum factor");
		txtMomentumFactor.setBounds(327, 76, 85, 20);
		getContentPane().add(txtMomentumFactor);
		txtMomentumFactor.setColumns(10);

		JRadioButton rdbtnDeterministic = new JRadioButton("Deterministic");
		rdbtnDeterministic.setBounds(155, 75, 109, 23);
		getContentPane().add(rdbtnDeterministic);

		spinnerEpochSize.setModel(new SpinnerNumberModel(10000, 1, 48000, 100));
		spinnerEpochSize.setBounds(239, 128, 71, 20);
		getContentPane().add(spinnerEpochSize);

		JButton btnLaunch = new JButton("Launch Learning Algorithm");
		btnLaunch.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {

				btnLaunch.setEnabled(false);

				learningAlg((simpleBox.getSelectedIndex() == 1), rdbtnAdaptableLearningRate.isSelected(),
						rdbtnPreprocessing.isSelected(), !rdbtnIncrementEveryTime.isSelected(),
						(double) spinnerLR.getValue(), (double) spinnerDecreaseLR.getValue(),
						(double) spinnerIncreaseLR.getValue(), (double) spinnerMomentumFact.getValue(),
						(double) spinnerLearningProp.getValue(), (double) spinnerTarget.getValue(),
						(int) spinnerEpochSize.getValue(), (int) spinnerRefreshRate.getValue(),
						(boolean) rdbtnDeterministic.isSelected());

			}
		});
		btnLaunch.setBounds(10, 159, 464, 23);
		getRootPane().setDefaultButton(btnLaunch);
		getContentPane().add(btnLaunch);

		txtEpochSize = new JTextField();
		txtEpochSize.setText("Epoch Size");
		txtEpochSize.setOpaque(false);
		txtEpochSize.setHorizontalAlignment(SwingConstants.TRAILING);
		txtEpochSize.setFocusable(false);
		txtEpochSize.setEditable(false);
		txtEpochSize.setColumns(10);
		txtEpochSize.setBorder(null);
		txtEpochSize.setBounds(144, 128, 85, 20);
		getContentPane().add(txtEpochSize);

		txtRefreshRate = new JTextField();
		txtRefreshRate.setText("Sample Size");
		txtRefreshRate.setOpaque(false);
		txtRefreshRate.setHorizontalAlignment(SwingConstants.TRAILING);
		txtRefreshRate.setFocusable(false);
		txtRefreshRate.setEditable(false);
		txtRefreshRate.setColumns(10);
		txtRefreshRate.setBorder(null);
		txtRefreshRate.setBounds(309, 127, 85, 20);
		getContentPane().add(txtRefreshRate);

		JSpinner spinnerNumberOfSamples = new JSpinner();
		spinnerNumberOfSamples.setModel(new SpinnerNumberModel(new Integer(40), new Integer(1), null, new Integer(1)));
		spinnerNumberOfSamples.setBounds(923, 22, 50, 20);
		getContentPane().add(spinnerNumberOfSamples);

		txtNumberOfSamples = new JTextField();
		txtNumberOfSamples.setText("Number of samples per algorithm");
		txtNumberOfSamples.setOpaque(false);
		txtNumberOfSamples.setHorizontalAlignment(SwingConstants.TRAILING);
		txtNumberOfSamples.setFocusable(false);
		txtNumberOfSamples.setEditable(false);
		txtNumberOfSamples.setColumns(10);
		txtNumberOfSamples.setBorder(null);
		txtNumberOfSamples.setBounds(743, 22, 170, 20);
		getContentPane().add(txtNumberOfSamples);

		JSpinner spinnerNumberOfAlg = new JSpinner();
		spinnerNumberOfAlg.setModel(new SpinnerNumberModel(new Integer(10), new Integer(1), null, new Integer(1)));
		spinnerNumberOfAlg.setBounds(923, 50, 50, 20);
		getContentPane().add(spinnerNumberOfAlg);

		txtNumberOfAlgorithm = new JTextField();
		txtNumberOfAlgorithm.setText("Number of Algorithm Iterations");
		txtNumberOfAlgorithm.setOpaque(false);
		txtNumberOfAlgorithm.setHorizontalAlignment(SwingConstants.TRAILING);
		txtNumberOfAlgorithm.setFocusable(false);
		txtNumberOfAlgorithm.setEditable(false);
		txtNumberOfAlgorithm.setColumns(10);
		txtNumberOfAlgorithm.setBorder(null);
		txtNumberOfAlgorithm.setBounds(760, 50, 153, 20);
		getContentPane().add(txtNumberOfAlgorithm);

		JButton btnLaunchMultipleLearning = new JButton("Launch Multiple Learning Algorithm");
		btnLaunchMultipleLearning.setBounds(509, 159, 423, 23);
		btnLaunchMultipleLearning.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {

				btnLaunchMultipleLearning.setEnabled(false);

				massLearningAlg((simpleBox.getSelectedIndex() == 1), rdbtnAdaptableLearningRate.isSelected(),
						rdbtnPreprocessing.isSelected(), !rdbtnIncrementEveryTime.isSelected(),
						(double) spinnerLR.getValue(), (double) spinnerDecreaseLR.getValue(),
						(double) spinnerIncreaseLR.getValue(), (double) spinnerMomentumFact.getValue(),
						(double) spinnerLearningProp.getValue(), (int) spinnerNumberOfSamples.getValue(),
						(int) spinnerEpochSize.getValue(), (int) spinnerRefreshRate.getValue(),
						(int) spinnerNumberOfAlg.getValue());

			}
		});
		getContentPane().add(btnLaunchMultipleLearning);

		iterationTxt = new JTextField();
		iterationTxt.setText("0");
		iterationTxt.setOpaque(false);
		iterationTxt.setHorizontalAlignment(SwingConstants.TRAILING);
		iterationTxt.setFocusable(false);
		iterationTxt.setEditable(false);
		iterationTxt.setColumns(10);
		iterationTxt.setBorder(null);
		iterationTxt.setBounds(942, 160, 31, 20);
		getContentPane().add(iterationTxt);

		// contentPane.setBackground(Color.ORANGE);
		// setContentPane(contentPane);
		setVisible(true);

	}

	public static double[] createInput(int imageNumber, byte[] imagesArray) {
		double[] res = new double[784];
		;
		for (int i = 0; i < 28 * 28; i++) {
			res[i] = (imagesArray[16 + imageNumber * (28 * 28) + i]) / 128.;
		}
		return res;
	}

	/**
	 * Creates an <code>ArrayList</code> of <code>SourceImage</code>s following
	 * two byte arrays.
	 * 
	 * @param rawImagesArray
	 * @param labelsArray
	 * @return
	 */
	public static List<SourceImage> createCleanInput(byte[] rawImagesArray, byte[] labelsArray) {
		List<SourceImage> res = new ArrayList<SourceImage>();
		int imageSize = 28;
		int init = 16;
		int c = 0;

		while (init + (imageSize * imageSize) * c < rawImagesArray.length) {
			byte[] temp = Arrays.copyOfRange(rawImagesArray, init + (imageSize * imageSize * c),
					init + imageSize * imageSize * (c + 1));

			res.add(new SourceImage(temp, imageSize, labelsArray[8 + c]));
			c++;
		}
		return res;
	}

	/**
	 * Recreates an <code>ArrayList</code> of <code>SourceImage</code>s
	 * following the processed data from features and expectedOutputs. As such
	 * the SourceImages do not actually have the image code, only the extracted
	 * features.
	 * 
	 * @param features
	 * @param expectedOutputs
	 * @return
	 */
	private static List<SourceImage> recreateCleanInput(double[][] features, double[][] expectedOutputs) {

		List<SourceImage> res = new ArrayList<SourceImage>();
		for (int i = 0; i < features.length; i++) {
			res.add(new SourceImage(features[i], expectedOutputs[i]));
		}

		return res;
	}

	public static int maxIndex(List<Double> list) {
		int res = 0;
		double maxValue = Double.MIN_VALUE;

		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) > maxValue) {
				res = i;
				maxValue = list.get(i);
			}
		}
		return res;
	}

	public static int maxIndex(double[] array) {
		int res = 0;
		double maxValue = Double.MIN_VALUE;

		for (int i = 0; i < array.length; i++) {
			if (array[i] > maxValue) {
				res = i;
				maxValue = array[i];
			}
		}
		return res;
	}

	private static int[] getRandomPermutation(int size) {
		int[] res = new int[size];
		int temp;
		int tempI;
		for (int i = 0; i < size; i++) {
			res[i] = i;
		}
		for (int i = 0; i < size; i++) {

			tempI = (int) Math.random() * size;
			temp = res[tempI];
			res[tempI] = res[i];
			res[i] = temp;
		}
		return res;
	}

	private static List<SourceImage> permutateRandomly(List<SourceImage> cleanInput) {
		int[] index = getRandomPermutation(cleanInput.size());
		List<SourceImage> res = new ArrayList<SourceImage>();
		for (int i = 0; i < cleanInput.size(); i++) {
			res.add(cleanInput.get(index[i]));
		}
		return res;

	}

	private double testLearningNN(int totalSourceSize, int learningSourceSize, List<SourceImage> cleanInput,
			NeuralNetwork learningNN, boolean preprocessing, int sampleNumber, int sampleSize, Output out,
			int[] currentPermutation) {
		double[] input;
		double averageTestMistakes = 0;
		double averageLearningMistakes = 0;
		double averageTestError = 0;
		double averageLearningError = 0;

		for (int j = 0; j < learningSourceSize; j++) {
			SourceImage currentImage = cleanInput.get(currentPermutation[j]);
			if (preprocessing) {
				input = currentImage.getRelevantFeatures();
			} else {
				input = currentImage.getCleanRawDoubleImage();
			}
			learningNN.setInputs(input);
			learningNN.fire();
			double[] expectedOutput = currentImage.getExpectedOutput();
			if (maxIndex(learningNN.getOutputs()) != maxIndex(currentImage.getExpectedOutput())) {
				averageLearningMistakes++;
			}

			averageLearningError += currentError(expectedOutput, learningNN.getOutputs());
		}
		averageLearningMistakes = averageLearningMistakes / learningSourceSize;
		averageLearningError = averageLearningError / learningSourceSize;

		for (int j = learningSourceSize; j < totalSourceSize; j++) {
			SourceImage currentImage = cleanInput.get(j);
			if (preprocessing) {
				input = currentImage.getRelevantFeatures();
			} else {
				input = currentImage.getCleanRawDoubleImage();
			}
			learningNN.setInputs(input);
			learningNN.fire();
			double[] expectedOutput = currentImage.getExpectedOutput();
			if (maxIndex(learningNN.getOutputs()) != maxIndex(currentImage.getExpectedOutput())) {
				averageTestMistakes++;
			}

			averageTestError += currentError(expectedOutput, learningNN.getOutputs());
		}
		averageTestError = averageTestError / (totalSourceSize - learningSourceSize);
		averageTestMistakes = (averageTestMistakes) / (totalSourceSize - learningSourceSize);

		// System.out.println("Test error :" +averageTestError*100 +"Learning
		// Error : " + averageLearningError*100);

		out.getNumberOfExample().add((double) sampleNumber * sampleSize);
		testErrorSeries.add((double) sampleNumber, averageTestError * 100);
		out.getAverageQuadraticErrorTest().add(averageTestError * 100);
		errorSeries.add((double) sampleNumber, averageLearningError * 100);
		out.getAverageQuadraticErrorLearning().add(averageLearningError * 100);
		testMistakeSeries.add((double) sampleNumber, averageTestMistakes * 100);
		out.getMistakePerEpochTest().add(averageTestMistakes * 100);
		mistakeSeries.add((double) sampleNumber, averageLearningMistakes * 100);
		out.getMistakePerEpochLearning().add(averageLearningMistakes * 100);

		return averageTestError * 100;
	}
}
