#pragma once

struct target {
	int x;
	int y;
	int width;
	int height;
	int directx;
	int directy;
	int uuid;
	int life;
	int dist;
	bool b_new;
	float score;
	target* prev;

	target() :
		x(0),
		y(0),
		width(0),
		height(0),
		directx(0),
		directy(0),
		uuid(0),
		life(1),
		dist(1e4),
		b_new(true),
		score(0.0),
		prev(nullptr) {}
};