package com.nakaji.shimokita.extention;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nakaji.shimokita.core.FeatureVector;

public class DefaultFeatureVector implements FeatureVector {
	private List<? extends Object> objects;

	public DefaultFeatureVector(List<? extends Object> objects) {
		this.objects = objects;
	}

	@Override
	public Map<Object, Double> getElements() {
		Map<Object, Double> result = new HashMap<Object, Double>();
		for (Object o : objects) {
			result.put(o, 1.0);
		}
		return result;
	}

}
