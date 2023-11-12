#pragma once

#include "Colorer.h"

class IncidenceColorer : public Colorer
{
public:
	static Coloring* color(Graph& graph);
};