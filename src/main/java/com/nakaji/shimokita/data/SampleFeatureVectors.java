package com.nakaji.shimokita.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nakaji.shimokita.core.FeatureVector;
import com.nakaji.shimokita.extention.DefaultFeatureVector;

public class SampleFeatureVectors {
	private SampleFeatureVectors() {
	};

	public static Map<String, List<FeatureVector>> generate() {
		Map<String, List<FeatureVector>> result = new HashMap<String, List<FeatureVector>>();
		List<FeatureVector> elders = new ArrayList<FeatureVector>();
		elders.add(new DefaultFeatureVector(Arrays.asList("misora", "pinklady", "omochi")));
		elders.add(new DefaultFeatureVector(Arrays.asList("hagimoto", "pinklady", "kurosawa")));
		List<FeatureVector> youngers = new ArrayList<FeatureVector>();
		youngers.add(new DefaultFeatureVector(Arrays.asList("pokemon", "omochi", "gakki")));
		youngers.add(new DefaultFeatureVector(Arrays.asList("pokemon", "monsto", "gakki")));
		result.put("elders", elders);
		result.put("youngers", youngers);
		return result;
	}
}
