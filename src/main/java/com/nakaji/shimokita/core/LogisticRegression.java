package com.nakaji.shimokita.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class LogisticRegression implements Trainer {
	private double initialEmpiricalParameter;
	private double l2 = 1.0;

	public LogisticRegression(double initialEmpiricalParameter) {
		this.initialEmpiricalParameter = initialEmpiricalParameter;
	}

	@Override
	public <L> Classifier<L> train(Map<L, List<FeatureVector>> featureMap) {
		List<L> labels = labels(featureMap);
		List<LabelFeatureVector<L>> allFeatureVectors = extractAllFeatureVectors(featureMap);
		Map<L, Map<Object, Double>> weights = initializeWeight(allFeatureVectors, labels);
		Map<L, Map<Object, Double>> historicalGradientSquares = new HashMap<L, Map<Object, Double>>();
		for (int i = 0; i < 100; i++) {
			for (LabelFeatureVector<L> f : allFeatureVectors) {
				update(weights, f.label, f.vector, historicalGradientSquares);
			}
			System.out.println(weights);
		}
		LinearClassifier<L> result = new LinearClassifier<L>();
		//result.setBiases(biases);
		//result.setWeights(weights);
		return result;
	}

	private <L> void update(Map<L, Map<Object, Double>> weights, L rightLabel, FeatureVector f,
			Map<L, Map<Object, Double>> historicalGradientSquares) {
		Set<L> labels = weights.keySet();
		Map<L, Double> probabilities = new HashMap<L, Double>();
		double denominator = 0;
		for (L label : labels) {
			Map<Object, Double> weight = weights.get(label);
			double probability = weight.get(Bias.class);
			for (Entry<Object, Double> e : f.getElements().entrySet()) {
				Object feature = e.getKey();
				probability += Math.exp(e.getValue() * weight.get(feature));
			}
			probabilities.put(label, probability);
			denominator += probability;
		}
		for (L label : labels) {
			double y = probabilities.get(label) / denominator;
			double magicValue = rightLabel.equals(label) ? (1 - y) : -y;
			Map<Object, Double> weight = weights.get(label);
			double gradient = magicValue - l2 * weight.get(Bias.class);
			Map<Object, Double> gradientSquare = historicalGradientSquares.get(label);
			if (gradientSquare == null) {
				gradientSquare = new HashMap<Object, Double>();
				historicalGradientSquares.put(label, gradientSquare);
			}
			updateGradientSquare(Bias.class, gradientSquare, magicValue);
			weight.put(Bias.class, weight.get(Bias.class) + initialEmpiricalParameter * gradient
					/ Math.sqrt(gradientSquare.get(Bias.class)));
			for (Entry<Object, Double> e : f.getElements().entrySet()) {
				Object feature = e.getKey();
				gradient = magicValue * e.getValue() - l2 * weight.get(feature);
				updateGradientSquare(feature, gradientSquare, magicValue * e.getValue());
				weight.put(feature, weight.get(feature) + initialEmpiricalParameter * gradient
						/ Math.sqrt(gradientSquare.get(feature)));
			}
		}
	}

	private <L> double updateGradientSquare(Object key, Map<Object, Double> gradientSquares,
			Double gradient) {
		Double previous = gradientSquares.get(key);
		if (previous == null) {
			double result = 0.01 + gradient * gradient;
			gradientSquares.put(key, result);
			return gradient * gradient;
		}
		double result = (previous + gradient) * (previous + gradient);
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

	private static class Bias {

	}
}
