#pragma once

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

#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <utility>

// Ensure.hpp is an optional dependency: if it's available (either as part of this
// checkout or vendored alongside this header), use its ensure() for a formatted
// diagnostic on failure; otherwise fall back to plain assert() so this header still
// works standalone.
#if __has_include("commons/Ensure.hpp")
	#include "commons/Ensure.hpp"
#elif __has_include("Ensure.hpp")
	#include "Ensure.hpp"
#else
	#include <cassert>
	// Guard against Ensure.hpp having already been included under a path our
	// __has_include checks above don't know about (e.g. vendored elsewhere as
	// "3rdparty/Ensure.hpp") -- COMMONS_ENSURE_HPP is defined by Ensure.hpp
	// itself, so this still catches that case even under an unknown filename.
	#if !defined(COMMONS_ENSURE_HPP) && !defined(ensure)
		#define ensure(condition, ...) assert((condition))
	#endif
#endif

#include "Flattener.hpp"

typedef std::vector<unsigned int> FlattenerShape;

/**
 * @brief Tracks which points of a large multi-dimensional parameter space have been marked,
 * compressing whole axis-aligned "hyper-rows" into lower-dimensional zones as they fill up.
 *
 * Construct a TensorMask over a shape (the size of each dimension, outermost first -- the same
 * convention as Flattener), then add() individual flat indices as they're determined to be
 * marked, and contains() to test whether a given flat index has been marked. Internally, whenever
 * every point along one dimension of a zone becomes marked, that information is collapsed into a
 * lower-dimensional zone spanning the remaining dimensions instead of being stored point-by-point
 * -- so a marked region that aligns with the axes of the space can be tracked in far less memory
 * than the size of the space itself, even for spaces far larger than could be stored point-by-
 * point. A highly irregular marked region that doesn't align with any axis degrades toward one
 * entry per marked point.
 *
 * configure() narrows add()/contains()/index()/flatten()/size() to operate against a subset of
 * dimensions rather than the full space; switching configuration never discards anything marked
 * under a previous configuration, at any granularity. Within whichever shape is currently
 * configured, flat indices must be add()ed in ascending order -- add() aborts via ensure() if
 * this is violated.
 *
 * @tparam I Type used for flat indices and the total element count of the configured space.
 * Defaults to uint64_t; must be able to represent the flattened size of whichever dimension
 * subset is configured.
 */
template<typename I=std::uint64_t>
class TensorMask {
public:
	/**
	 * @brief Constructs a TensorMask for the given shape, with nothing marked.
	 *
	 * @param shape The size of each dimension, outermost first. At most 64 dimensions are
	 * supported, since a dimension subset is tracked internally as a 64-bit bitmask.
	 */
	TensorMask(const FlattenerShape& shape) : shape(shape) {
		reset();
	}

	/**
	 * @brief Constructs a TensorMask for the given shape, with nothing marked.
	 *
	 * Equivalent to the std::vector constructor; provided so a shape can be written inline,
	 * e.g. `TensorMask<> mask({2, 3, 4});`.
	 *
	 * @param shape The size of each dimension, outermost first.
	 */
	TensorMask(std::initializer_list<unsigned int> shape) : shape(shape) {
		reset();
	}

	/**
	 * @brief Clears every mark and returns to the default configuration (every dimension
	 * active, in the order passed to the constructor).
	 */
	void reset() {
		isFull = false;
		zones.clear();

		std::vector<unsigned int> indices(shape.size());
		std::iota(indices.begin(), indices.end(), 0);
		configure(indices);
	}

	/**
	 * @brief Returns whether every point in the currently configured space has been marked.
	 */
	bool full() const {
		return isFull;
	}

	/**
	 * @brief Returns whether no point anywhere in the space has been marked, independent of
	 * the currently configure()d view.
	 */
	bool empty() const {
		for(const auto& pair : zones)
			if(!pair.second.contents.empty())
				return false;
		return true;
	}

	/**
	 * @brief Narrows add()/contains()/index()/flatten()/size() to operate over a subset of the
	 * shape's dimensions instead of the full space.
	 *
	 * Marks recorded under any past configuration remain intact and visible through this (or any
	 * future) configuration -- configure() only changes which dimensions add()/contains() etc.
	 * address, never what has already been marked.
	 *
	 * @param parameterIndices The dimensions to make active, as indices into the shape passed to
	 * the constructor, in strictly ascending order. May be a proper subset of all dimensions.
	 */
	void configure(const std::vector<unsigned int>& parameterIndices) {
		configuredIndices = parameterIndices;

		Bitmap configuredBitmap = indicesToBitmap(configuredIndices.begin(), configuredIndices.end());
		configuredZone = &ensureZone(configuredBitmap);

		searchOrder.clear();
		configuredZones.clear();
		for(auto& pair : zones)
			if((pair.first & configuredBitmap) == pair.first)
				configureSubzone(pair.second);
	}

	/**
	 * @brief Returns the coordinate of one dimension of the configured space for a given flat
	 * index. Specific to the currently configured zone.
	 *
	 * @param dimension Which of the configured dimensions to extract, indexed the same way as
	 * the parameterIndices passed to configure() (0 is the first configured dimension).
	 * @param flatIndex A flat index within the configured space, in `[0, size())`.
	 * @return The coordinate of @p dimension corresponding to @p flatIndex.
	 */
	inline unsigned int index(unsigned int dimension, unsigned int flatIndex) const {
		ensure(configuredZone != nullptr);
		return configuredZone->flattener.index(dimension, flatIndex);
	}

	/**
	 * @brief Returns the flat index for the given coordinates within the configured space.
	 * Specific to the currently configured zone.
	 *
	 * @param coordinates One coordinate per configured dimension, in the same order as the
	 * parameterIndices passed to configure().
	 * @return The flat index corresponding to @p coordinates.
	 */
	inline unsigned int flatten(const std::vector<unsigned int>& coordinates) const {
		ensure(configuredZone != nullptr);
		return configuredZone->flattener.flatten(coordinates);
	}

	/**
	 * @brief Returns the total number of flat indices in the currently configured space, i.e.
	 * the product of the configured dimensions' sizes.
	 */
	inline I size() const {
		return configuredZone->flattener.size();
	}

	/**
	 * @brief Returns whether the given flat index, within the currently configured space, has
	 * been marked -- whether directly, or because some lower-dimensional zone it collapsed into
	 * covers it.
	 *
	 * @param index A flat index within the configured space, in `[0, size())`.
	 */
	bool contains(I index) {
		if(isFull) {
			return true;
		} else {
			std::vector<unsigned int> subspaceIndices;
			for(int searchIndex = searchOrder.size() - 1; searchIndex >= 0 ; --searchIndex) {
				const int configuredZoneIndex = searchOrder[searchIndex];
				const ConfiguredSubzone& subzone = configuredZones[configuredZoneIndex];

				subspaceIndices.clear();
				for(int map : subzone.map)
					subspaceIndices.push_back(configuredZone->flattener.index(map, index));
				I subspaceIndex = subzone.zone.flattener.flatten(subspaceIndices);

				if(subzone.zone.contents.contains(subspaceIndex)) {
					// Move the containing subzone to the back of searchOrder so it's tried first next time
					std::swap(searchOrder[searchIndex], searchOrder.back());
					return true;
				}
			}
		}
		return false;
	}

	/**
	 * @brief Marks the given flat index within the currently configured space.
	 *
	 * Indices must be added in strictly ascending order within whichever shape is currently
	 * configured; this is checked via ensure(), which aborts the process on violation.
	 *
	 * @param index A flat index within the configured space, in `[0, size())`.
	 */
	void add(I index) {
		if(!isFull)
			add(index, configuredZone);
	}

	/// The size of each dimension of the full space, outermost first, as passed to the constructor.
	const std::vector<std::uint32_t> shape;

private:
	typedef std::uint64_t Bitmap;	// Allows for 64 parameters max

	struct Zone {
		Zone(Bitmap bitmap, const FlattenerShape& shape) : bitmap(bitmap), flattener(shape), dimensions(shape.size()) {}

		const Bitmap bitmap;
		Flattener<I> flattener;
		std::unordered_set<I> contents;
		I lastAdded = 0;
		unsigned int dimensions;
	};

	struct ConfiguredSubzone {
		ConfiguredSubzone(Zone& zone, std::vector<unsigned int> map) : zone(zone), map(map) {}
		Zone& zone;
		const std::vector<unsigned int> map;
	};

	inline unsigned int configuredIndexOf(unsigned int target) const {
		for(unsigned int index = 0; index < configuredIndices.size(); ++index) {
			if(configuredIndices[index] == target)
				return index;
		}
		ensure(false);
		return 0;
	}

	void configureSubzone(Zone& zone) {
		std::vector<unsigned int> subspaceIndices = bitmapToIndices(zone.bitmap);
		std::vector<unsigned int> map;
		for(unsigned int subspaceIndex : subspaceIndices)
			map.push_back(configuredIndexOf(subspaceIndex));
		searchOrder.push_back(configuredZones.size());
		configuredZones.emplace_back(zone, map);
	}

	Zone& ensureZone(Bitmap bitmap) {
		if(!zones.contains(bitmap)) {
			std::vector<unsigned int> indices = bitmapToIndices(bitmap);
			FlattenerShape zoneShape;
			for(unsigned int index : indices)
				zoneShape.push_back(shape[index]);
			zones.emplace(bitmap, std::move(Zone(bitmap, zoneShape)));
		}
		return zones.at(bitmap);
	}

	template<class It>
	I indicesToBitmap(It begin, It end) const {
		Bitmap bitmap = 0;
		int lastIndex = -1;
		for(It iterator = begin; iterator != end; ++iterator) {
			ensure(static_cast<int>(*iterator) > lastIndex);		// Ascending order
			ensure(*iterator < shape.size());	// Valid range
			lastIndex = *iterator;

			bitmap |= 1L << *iterator;
		}
		return bitmap;
	}

	std::vector<unsigned int> bitmapToIndices(Bitmap bitmap) const {
		std::vector<unsigned int> indices;
		unsigned int index = 0;
		while(bitmap != 0) {
			if((bitmap & 1) != 0)
				indices.push_back(index);
			index++;
			bitmap >>= 1;
		}
		return indices;
	}

	void add(I index, Zone* zone, bool alwaysCheckRows = false) {
		ensure(zone->contents.empty() || zone->lastAdded <= index);
		zone->contents.insert(index);
		zone->lastAdded = index;

		// Check each dimension to see whether it has "filled up," and can be summarized
		// in a lower dimensional space.
		const std::vector<unsigned int> bitmapIndices = bitmapToIndices(zone->bitmap);
		for(unsigned int dimension = 0; dimension < zone->dimensions; ++dimension) {
			// Check whether we are at the end of a hyper-row in this dimension
			const unsigned int dimensionIndex = zone->flattener.index(dimension, index);
			const unsigned int dimensionSize = shape[bitmapIndices[dimension]];

			if(dimensionIndex == dimensionSize - 1 || alwaysCheckRows) {
				// We are at the end of a hyper-row. Check whether the entire hyper-row is marked

				// First reconstruct the parameter vector for this index
				const std::vector<unsigned int> parameterVector = zone->flattener.indices(index);
				std::vector<unsigned int> testVector = parameterVector;
				bool allMarked = true;
				for(unsigned int coordinate = 0; coordinate < dimensionSize && allMarked; ++coordinate) {
					testVector[dimension] = coordinate;
					const I rowIndex = zone->flattener.flatten(testVector);
					allMarked &= zone->contents.contains(rowIndex);
				}

				if(allMarked) {
					// This entire hyper-row is marked
					if(zone->dimensions == 1) {
						// The hyper-row is only one dimension, so the entire parameter space must be marked
						isFull = true;
					} else {
						// Shift the marking responsibility from this zone to a lower-dimensional one

						// Compute the bitmapIndices for the maybe-new subzone
						std::vector<unsigned int> subzoneIndices = bitmapIndices;
						subzoneIndices.erase(subzoneIndices.begin() + dimension);
						const I subzoneBitmap = indicesToBitmap(subzoneIndices.begin(), subzoneIndices.end());

						// Get a reference to the subzone we are moving the marked information to
						const bool addToConfiguration = !zones.contains(subzoneBitmap);
						Zone& subzone = ensureZone(subzoneBitmap);
						if(addToConfiguration)
							configureSubzone(subzone);

						// Compute the parameters that we are going to add to the subzone.  Should be the same
						// as indices, but with one dimension removed
						std::vector<unsigned int> subzoneParameterVector = parameterVector;
						subzoneParameterVector.erase(subzoneParameterVector.begin() + dimension);

						// Finally, get the subzone index, and add the mark to it
						const I subzoneIndex = subzone.flattener.flatten(subzoneParameterVector);
						add(subzoneIndex, &subzone, true);

						// OPTIONAL: Remove equivalent marks from the configuredZone to save space
						for(unsigned int coordinate = 0; coordinate < dimensionSize; ++coordinate) {
							testVector[dimension] = coordinate;
							const I cellIndex = zone->flattener.flatten(testVector);
							zone->contents.erase(cellIndex);
						}
					}
				}
			}
		}
	}

	// Current configuration
	Zone* configuredZone;
	std::vector<unsigned int> configuredIndices;
	std::vector<ConfiguredSubzone> configuredZones;
	std::vector<unsigned int> searchOrder;

	// Mask zones
	std::unordered_map<std::uint64_t /* indices bitmap */, Zone> zones;
	bool isFull = false;
};
