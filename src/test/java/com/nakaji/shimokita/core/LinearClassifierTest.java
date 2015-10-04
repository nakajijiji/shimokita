package com.nakaji.shimokita.core;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

public class LinearClassifierTest {
	@Test
	public void testClassify() {
		LinearClassifier<String> classifier = new LinearClassifier<String>();
		Map<String, Double> biases = new HashMap<String, Double>();
		biases.put("positive", 0.1);
		biases.put("negative", -0.1);
		Map<String, Map<Object, Double>> weights = new HashMap<String, Map<Object, Double>>();
		Map<Object, Double> positiveWeights = new HashMap<Object, Double>();
		positiveWeights.put("feature1", 0.3);
		weights.put("positive", positiveWeights);
		Map<Object, Double> negativeWeights = new HashMap<Object, Double>();
		negativeWeights.put("feature1", -0.3);
		weights.put("positive", negativeWeights);
		classifier.setBiases(biases);
		classifier.setWeights(weights);

		FeatureVector vector = new FeatureVector() {
			@Override
			public Map<Object, Double> getElements() {
				Map<Object, Double> result = new HashMap<Object, Double>();
				result.put("feature1", 0.1);
				result.put("feature2", 0.3);
				return result;
			}
		};
		assertEquals("positive", classifier.classify(vector));
	}
}
