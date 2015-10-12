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

	public LogisticRegression(double initialEmpiricalParameter) {
		this.initialEmpiricalParameter = initialEmpiricalParameter;
	}

	@Override
	public <L> Classifier<L> train(Map<L, List<FeatureVector>> featureMap) {
		List<L> labels = labels(featureMap);
		Map<L, Double> biases = initializeBias(labels);
		List<LabelFeatureVector<L>> allFeatureVectors = extractAllFeatureVectors(featureMap);
		Map<L, Map<Object, Double>> weights = initializeWeight(allFeatureVectors, labels);
		Map<L, Map<Object, Double>> historicalGradientSquares = new HashMap<L, Map<Object, Double>>();
		for (int i = 0; i < 100; i++) {
			for (LabelFeatureVector<L> f : allFeatureVectors) {
				update(biases, weights, f.label, f.vector, historicalGradientSquares);
			}
		}
		LinearClassifier<L> result = new LinearClassifier<L>();
		result.setBiases(biases);
		result.setWeights(weights);
		return result;
	}

	private <L> void update(Map<L, Double> biases, Map<L, Map<Object, Double>> weights,
			L rightLabel, FeatureVector f, Map<L, Map<Object, Double>> historicalGradientSquares) {
		Set<L> labels = biases.keySet();
		Map<L, Double> logits = new HashMap<L, Double>();
		double denominator = 0;
		for (L label : labels) {
			double logit = Math.exp(biases.get(label));
			Map<Object, Double> weight = weights.get(label);
			for (Entry<Object, Double> e : f.getElements().entrySet()) {
				Object feature = e.getKey();
				logit += Math.exp(e.getValue() * weight.get(feature));
			}
			logits.put(label, logit);
			denominator += logit;
		}
		// denominator
		for (L label : labels) {
			double y = logits.get(label) / denominator;
			double magicValue = rightLabel.equals(label) ? (1 - y) : -y;
			Map<Object, Double> gradientSquare = historicalGradientSquares.get(label);
			if (gradientSquare == null) {
				gradientSquare = new HashMap<Object, Double>();
				historicalGradientSquares.put(label, gradientSquare);
			}
			updateGradientSquare("__BIAS__", gradientSquare, magicValue);
			biases.put(
					label,
					biases.get(label) + initialEmpiricalParameter * magicValue
							/ Math.sqrt(gradientSquare.get("__BIAS__")));
			Map<Object, Double> weight = weights.get(label);
			for (Entry<Object, Double> e : f.getElements().entrySet()) {
				Object feature = e.getKey();
				double gradient = magicValue * e.getValue();
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

	private <L> Map<L, Double> initializeBias(List<L> labels) {
		Map<L, Double> result = new HashMap<L, Double>();
		for (L l : labels) {
			result.put(l, 0.0);
		}
		return result;
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
			results.put(label, result);
		}
		return results;
	}

	private static class LabelFeatureVector<L> {
		private FeatureVector vector;
		private L label;
	}
}
