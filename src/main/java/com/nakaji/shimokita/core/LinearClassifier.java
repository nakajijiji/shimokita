package com.nakaji.shimokita.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class LinearClassifier<L> implements Classifier<L> {
	private Map<L, Double> biases;
	private Map<L, Map<Object, Double>> weights;

	@Override
	public L classify(FeatureVector feature) {
		List<Score<L>> results = classifyWithDetail(feature);
		if (results.size() == 0) {
			return null;
		}
		return results.get(0).getLabel();
	}

	public List<Score<L>> classifyWithDetail(FeatureVector feature) {
		Set<L> keys = biases.keySet();
		if (keys == null) {
			return Collections.emptyList();
		}
		List<Score<L>> results = new ArrayList<Score<L>>();
		for (L key : keys) {
			double score = biases.get(key);
			Map<Object, Double> ws = weights.get(key);
			for (Entry<Object, Double> e : feature.getElements().entrySet()) {
				score += e.getValue() * ws.get(e.getKey());
			}
			Score<L> result = new Score<L>();
			result.label = key;
			result.score = score;
			results.add(result);
		}
		Collections.sort(results, new Comparator<Score<L>>() {
			@Override
			public int compare(Score<L> arg0, Score<L> arg1) {
				return Double.compare(arg0.getScore(), arg1.getScore());
			}
		});
		return results;
	}

	public Map<L, Double> getBiases() {
		return biases;
	}

	public void setBiases(Map<L, Double> biases) {
		this.biases = biases;
	}

	public Map<L, Map<Object, Double>> getWeights() {
		return weights;
	}

	public void setWeights(Map<L, Map<Object, Double>> weights) {
		this.weights = weights;
	}

	public static class Score<L> {
		private L label;
		private double score;

		public L getLabel() {
			return label;
		}

		public void setLabel(L label) {
			this.label = label;
		}

		public double getScore() {
			return score;
		}

		public void setScore(double score) {
			this.score = score;
		}
	}

}
