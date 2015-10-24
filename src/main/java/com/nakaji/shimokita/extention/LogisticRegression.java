package com.nakaji.shimokita.extention;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.nakaji.shimokita.core.Classifier;
import com.nakaji.shimokita.core.FeatureVector;
import com.nakaji.shimokita.core.LinearClassifier;
import com.nakaji.shimokita.core.Trainer;

public class LogisticRegression implements Trainer {
	private double initialEmpiricalParameter = 1.0;
	private double l2 = 0.0;
	private double epsilon = 0.01;
	private int maxLoop = 1000;

	public LogisticRegression() {
	}

	public LogisticRegression withL2(double l2) {
		this.l2 = l2;
		return this;
	}

	public LogisticRegression withEpsilon(double epsilon) {
		this.epsilon = epsilon;
		return this;
	}

	public LogisticRegression withMaxLoop(int maxLoop) {
		this.maxLoop = maxLoop;
		return this;
	}

	public LogisticRegression withInitialEmpiricalParameter(double parameter) {
		this.initialEmpiricalParameter = parameter;
		return this;
	}

	@Override
	public <L> Classifier<L> train(Map<L, List<FeatureVector>> featureMap) {
		List<L> labels = labels(featureMap);
		List<LabelFeatureVector<L>> allFeatureVectors = extractAllFeatureVectors(featureMap);
		Map<L, Map<Object, Double>> weights = initializeWeight(
				allFeatureVectors, labels);
		Map<L, Map<Object, Double>> historicalGradientSquares = new HashMap<L, Map<Object, Double>>();
		double previousError = Integer.MAX_VALUE;
		for (int i = 0; i < maxLoop; i++) {
			double error = 0;
			for (LabelFeatureVector<L> f : allFeatureVectors) {
				error += update(weights, f.label, f.vector,
						historicalGradientSquares);
			}
			if (isConverged(previousError, error)) {
				break;
			}
		}
		return finalize(weights);
	}

	private boolean isConverged(double previousError, double currentError) {
		return Math.abs(previousError - currentError) < epsilon;
	}

	private <L> LinearClassifier<L> finalize(Map<L, Map<Object, Double>> weights) {
		LinearClassifier<L> result = new LinearClassifier<L>();
		Map<L, Double> biases = new HashMap<L, Double>();
		for (Entry<L, Map<Object, Double>> e : weights.entrySet()) {
			L label = e.getKey();
			Map<Object, Double> finalWeights = e.getValue();
			biases.put(label, finalWeights.get(Bias.class));
			finalWeights.remove(Bias.class);
		}
		result.setBiases(biases);
		result.setWeights(weights);
		return result;
	}

	private <L> double update(Map<L, Map<Object, Double>> weights,
			L rightLabel, FeatureVector f,
			Map<L, Map<Object, Double>> historicalGradientSquares) {
		Set<L> labels = weights.keySet();
		Map<L, Double> probabilities = new HashMap<L, Double>();
		double denominator = 0;
		for (L label : labels) {
			final DoubleObject confidence = new DoubleObject();
			final Map<Object, Double> weight = weights.get(label);
			forEachElementAndBias(f, new Procedure() {
				@Override
				protected void doProceed(Entry<Object, Double> e) {
					Object feature = e.getKey();
					confidence.value += e.getValue() * weight.get(feature);
				}
			});
			double probability = Math.exp(confidence.value);
			probabilities.put(label, probability);
			denominator += probability;
		}
		double error = 0;
		for (final L label : labels) {
			double y = probabilities.get(label) / denominator;
			final double labelError = rightLabel.equals(label) ? (1 - y) : -y;
			error += Math.abs(labelError);
			final Map<Object, Double> weight = weights.get(label);
			final Map<Object, Double> gradientSquare = historicalGradientSquares
					.containsKey(label) ? historicalGradientSquares.get(label)
					: new HashMap<Object, Double>();
			historicalGradientSquares.put(label, gradientSquare);
			forEachElementAndBias(f, new Procedure() {
				@Override
				protected void doProceed(Entry<Object, Double> e) {
					Object feature = e.getKey();
					double gradient = labelError * e.getValue() - l2
							* weight.get(feature);
					updateGradientSquare(feature, gradientSquare, gradient);
					weight.put(
							feature,
							weight.get(feature) + initialEmpiricalParameter
									* gradient
									/ Math.sqrt(gradientSquare.get(feature)));
				}
			});
		}
		return error;
	}

	private void forEachElementAndBias(FeatureVector f, Procedure procedure) {
		Entry<Object, Double> biasEntry = new Entry<Object, Double>() {
			@Override
			public Double setValue(Double value) {
				return null;
			}

			@Override
			public Double getValue() {
				return 1.0;
			}

			@Override
			public Object getKey() {
				return Bias.class;
			}
		};
		procedure.doProceed(biasEntry);
		for (Entry<Object, Double> e : f.getElements().entrySet()) {
			procedure.doProceed(e);
		}
	}

	private <L> double updateGradientSquare(Object key,
			Map<Object, Double> gradientSquares, Double gradient) {
		Double previous = gradientSquares.get(key);
		if (previous == null) {
			double result = 0.01 + gradient * gradient;
			gradientSquares.put(key, result);
			return gradient * gradient;
		}
		double result = previous + gradient * gradient;
		gradientSquares.put(key, result);
		return result;
	}

	private <L> List<L> labels(Map<L, List<FeatureVector>> featureMap) {
		return new ArrayList<L>(featureMap.keySet());
	}

	private <L> List<LabelFeatureVector<L>> extractAllFeatureVectors(
			Map<L, List<FeatureVector>> featureMap) {
		List<LabelFeatureVector<L>> result = new ArrayList<LogisticRegression.LabelFeatureVector<L>>();
		for (Entry<L, List<FeatureVector>> e : featureMap.entrySet()) {
			for (FeatureVector v : e.getValue()) {
				LabelFeatureVector<L> l = new LabelFeatureVector<L>();
				l.label = e.getKey();
				l.vector = v;
				result.add(l);
			}
		}
		return result;
	}

	private <L> Map<L, Map<Object, Double>> initializeWeight(
			List<LabelFeatureVector<L>> allFeatureVectors, List<L> labels) {
		Set<Object> allElements = new HashSet<Object>();
		for (LabelFeatureVector<L> f : allFeatureVectors) {
			for (Entry<Object, Double> e : f.vector.getElements().entrySet()) {
				allElements.add(e.getKey());
			}
		}
		Map<L, Map<Object, Double>> results = new HashMap<L, Map<Object, Double>>();
		for (L label : labels) {
			Map<Object, Double> result = new HashMap<Object, Double>();
			for (Object o : allElements) {
				result.put(o, 0.0);
			}
			result.put(Bias.class, 0.0);
			results.put(label, result);
		}
		return results;
	}

	private static class LabelFeatureVector<L> {
		private FeatureVector vector;
		private L label;
	}

	private static abstract class Procedure {
		protected abstract void doProceed(Entry<Object, Double> e);
	}

	private static class DoubleObject {
		private double value;
	}

	private static class Bias {
		@Override
		public String toString() {
			return "__BIAS__";
		}
	}
}
