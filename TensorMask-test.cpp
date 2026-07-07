/*
 * Copyright 2026 L. Richard Moore Jr.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <random>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TensorMask.hpp"

TEST_CASE( "No cells marked" ) {
	TensorMask<> mask({2, 3, 4});

	mask.configure({0, 1, 2});
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		CHECK(!isMarked);
	}
}

TEST_CASE( "One cell marked" ) {
	TensorMask<> mask({2, 3, 4});

	mask.configure({0, 1, 2});
	mask.add(0);
	const bool isMarked = mask.contains(0);
	CHECK(isMarked);

	for(unsigned int index = 1; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		CHECK(!isMarked);
	}
}

TEST_CASE( "One marked cell from subzone" ) {
	TensorMask<> mask({2, 3, 4});

	mask.configure({0, 1});
	mask.add(0);
	const bool isMarked = mask.contains(0);
	CHECK(isMarked);

	mask.configure({0, 1, 2});
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		if(mask.index(0, index) == 0 && mask.index(1, index) == 0)
			CHECK(isMarked);
		else
			CHECK(!isMarked);
	}
}

TEST_CASE( "One marked cell from two subzones" ) {
	TensorMask<> mask({2, 3, 4});

	mask.configure({0, 1});
	mask.add(0);
	bool isMarked = mask.contains(0);
	CHECK(isMarked);

	mask.configure({1, 2});
	mask.add(0);
	isMarked = mask.contains(0);
	CHECK(isMarked);

	mask.configure({0, 1, 2});
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		if((mask.index(0, index) == 0 && mask.index(1, index) == 0) || (mask.index(1, index) == 0 && mask.index(2, index) == 0))
			CHECK(isMarked);
		else
			CHECK(!isMarked);
	}
}

TEST_CASE( "One marked cell in superzone doesn't affect subzone" ) {
	TensorMask<> mask({2, 3, 4});

	mask.configure({0, 1, 2});
	mask.add(0);
	const bool isMarked = mask.contains(0);
	CHECK(isMarked);

	mask.configure({0, 1});
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		CHECK(!isMarked);
	}
}

TEST_CASE( "Subzone drawing from subberzone" ) {
	TensorMask<> mask({2, 3, 4});

	mask.configure({0});
	mask.add(0);
	mask.configure({0, 1});

	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		if(mask.index(0, index) == 0)
			CHECK(isMarked);
		else
			CHECK(!isMarked);
	}
}

TEST_CASE( "Default configuration + reset()" ) {
	TensorMask<> mask({2, 3, 4});
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		CHECK(!isMarked);
	}

	mask.configure({0});
	mask.add(0);
	mask.add(1);

	mask.configure({0,1,2});
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		CHECK(isMarked);
	}

	mask.reset();
	for(unsigned int index = 0; index < mask.size(); ++index) {
		const bool isMarked = mask.contains(index);
		CHECK(!isMarked);
	}
}

TEST_CASE( "Full row of marks migrates to subzone" ) {
	TensorMask<> mask({2, 2});

	mask.add(0);
	mask.add(1);
	CHECK(mask.contains(0));
	CHECK(mask.contains(1));
	CHECK(!mask.contains(2));
	CHECK(!mask.contains(3));
}

TEST_CASE( "Full mask" ) {
	TensorMask<> mask({2, 2});

	mask.add(0);
	mask.add(1);
	mask.add(2);
	mask.add(3);
	CHECK(mask.contains(0));
	CHECK(mask.contains(1));
	CHECK(mask.contains(2));
	CHECK(mask.contains(3));
	CHECK(mask.full());
}

TEST_CASE( "Full mask 2" ) {
	TensorMask<> mask({3, 3, 3});

	for(int index = 0; index < 27; index++) {
		CHECK(! mask.full());
		mask.add(index);

		for(int test = 0; test < 27; test++) {
			CAPTURE(index, test);
			if(test <= index)
				CHECK(mask.contains(test));
			else
				CHECK(!mask.contains(test));
		}
	}

	CHECK(mask.full());

	mask.reset();
	CHECK(! mask.full());

	// Confirm that it is now empty
	for(int index = 0; index < 27; index++) {
		CAPTURE(index);
		CHECK(!mask.contains(index));
	}
}

TEST_CASE( "Fuzz" ) {
	std::mt19937 generator(0);
	std::uniform_int_distribution<> countDistribution(0, 27);
	std::uniform_int_distribution<> indexDistribution(0, 26);

	for(int test = 0; test < 4096; ++test) {
		CAPTURE(test);
		TensorMask<> mask({3, 3, 3});

		const int count = countDistribution(generator);
		std::set<int> expectedSet;
		while(expectedSet.size() < static_cast<size_t>(count))
			expectedSet.insert(indexDistribution(generator));

		std::vector<unsigned int> expectedVector(expectedSet.begin(), expectedSet.end());
		expectedSet.clear();
		for(size_t i = 0; i < expectedVector.size(); ++i) {
			mask.add(expectedVector[i]);
			expectedSet.insert(expectedVector[i]);

			// Test
			for(int index = 0; index < 27; ++index) {
				CAPTURE(i, index);
				if(expectedSet.contains(index))
					CHECK(mask.contains(index));
				else
					CHECK(!mask.contains(index));
			}
		}

		if(expectedSet.size() == 27)
			CHECK(mask.full());
	}
}

TEST_CASE( "Exhaustive, high dimensions" ) {
	// In this space, there are 16 possible values.
	// We want to test every permutation of the 16 possible values.
	for(int permutation = 1; permutation < 65536; permutation++) {
		TensorMask<> mask({2, 2, 2, 2});
		std::set<unsigned int> expectedSet;

		int bit = 1;
		for(unsigned int index = 0; index < 16; ++index, bit <<= 1) {
			const bool add = permutation & bit;
			if(add) {
				expectedSet.insert(index);
				mask.add(index);
			}
		}

		for(int test = 0; test < 16; test++) {
			// CAPTURE(index, test);
			if(expectedSet.contains(test))
				CHECK(mask.contains(test));
			else
				CHECK(!mask.contains(test));
		}

		CHECK(mask.full() == (expectedSet.size() == 16));
	}
}

TEST_CASE( "Exhaustive, 2 dimensions" ) {
	// In this space, there are 16 possible values.
	// We want to test every permutation of the 16 possible values.
	for(int permutation = 1; permutation < 65536; permutation++) {
		TensorMask<> mask({4,4});
		std::set<unsigned int> expectedSet;

		int bit = 1;
		for(unsigned int index = 0; index < 16; ++index, bit <<= 1) {
			const bool add = permutation & bit;
			if(add) {
				expectedSet.insert(index);
				mask.add(index);
			}
		}

		for(int test = 0; test < 16; test++) {
			// CAPTURE(index, test);
			if(expectedSet.contains(test))
				CHECK(mask.contains(test));
			else
				CHECK(!mask.contains(test));
		}

		CHECK(mask.full() == (expectedSet.size() == 16));
	}
}

TEST_CASE( "empty()" ) {
	TensorMask<> mask({2, 3, 4});
	CHECK(mask.empty());

	mask.add(0);
	CHECK(!mask.empty());

	mask.reset();
	CHECK(mask.empty());
}

TEST_CASE( "empty() is unaffected by configure()ing to a new subspace with no add()" ) {
	TensorMask<> mask({2, 3, 4});
	CHECK(mask.empty());

	mask.configure({0, 1});
	CHECK(mask.empty());

	mask.configure({0, 1, 2});
	CHECK(mask.empty());
}

TEST_CASE( "flatten() round-trips with index()" ) {
	TensorMask<> mask({2, 3, 4});
	mask.configure({0, 1, 2});

	for(unsigned int a = 0; a < 2; ++a)
		for(unsigned int b = 0; b < 3; ++b)
			for(unsigned int c = 0; c < 4; ++c) {
				const unsigned int flat = mask.flatten({a, b, c});
				CHECK(mask.index(0, flat) == a);
				CHECK(mask.index(1, flat) == b);
				CHECK(mask.index(2, flat) == c);
			}
}
