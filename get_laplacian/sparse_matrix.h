#ifndef _SPARSE_MATRIX_H_
#define _SPARSE_MATRIX_H_

#include <memory>
#include <algorithm>

struct sparseEntry
{
	//sparseEntry(size_t _row, size_t _col, double _val) : row(_row), col(_col), value(_val)
	//{}

	size_t row;
	size_t col;
	double value;
	//
	//private: 
	//	sparseEntry() : row(0), col(0), value(NAN)
	//	{}
	bool operator<(sparseEntry &entry)
	{
		return col == entry.col ? row < entry.row : col < entry.col;
	}
};

int _cdecl compareSparseEntry(const void * const entry1, const void* const entry2)
{
	const sparseEntry* s1 = (sparseEntry*)entry1;
	const sparseEntry* s2 = (sparseEntry*)entry2;
	int colDiff = int(s1->col) - int(s2->col);
	int rowDiff = int(s1->row) - int(s2->row);
	return colDiff == 0 ? rowDiff : colDiff;
}


void fillSparseMatrix(sparseEntry * const entries, const size_t tlen, double * const vals, mwIndex * const row_inds, mwIndex * const col_inds, const long HW)
{
	std::qsort(&entries[0], tlen, sizeof(sparseEntry), compareSparseEntry);

	// current column index (goes from 0 to HW-1 and then HW)
	size_t col = 0;

	// current index in sorted values list
	size_t k = 0;

	// current index in the sparse matrix
	size_t s = 0;

	while (k < tlen && col < HW)
	{
		// process empty columns if any
		while (col < entries[k].col && col < HW)
		{
			// add empty column to sparse matrix
			col_inds[col] = s;
			//mexPrintf("\nColumn %d (%d): ", col, s);
			col++;
		}

		if (col == HW)
		{
			// last column was empty
			break;
		}

		// current column is non-empty
		col_inds[col] = s;
		//mexPrintf("\nColumn %d (%d): ", col, s);
		// process values, until they belong to the current column
		while (k < tlen && col == entries[k].col && col < HW)
		{
			double value = entries[k].value;
			size_t row = entries[k].row;
			k++;

			// there might be several values at the same row - add them together!
			while (k < tlen && col == entries[k].col && row == entries[k].row /*&& row < HW*/)
			{
				value += entries[k].value;
				k++;
			}
			if (k < tlen && entries[k].row < row && col == entries[k].col)
			{
				mexPrintf("Sorting Error!\n");
			}
			if (value != 0.0)
			{
				row_inds[s] = row;
				vals[s] = value;
				//mexPrintf("%d (%f), ", row, value);
				s++;
			}

		}
		col++;
	}
	while (col <= HW)
	{
		col_inds[col] = s;
		col++;
	}
}

#endif 