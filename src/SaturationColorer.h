#pragma once
#include "Colorer.h"

class SaturationColorer : public Colorer
{
public:
	static Coloring* color(Graph& graph);
};