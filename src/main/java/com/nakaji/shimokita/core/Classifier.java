package com.nakaji.shimokita.core;

import java.util.List;

public interface Classifier<L> {
	L classify(List<Feature> features);
}
