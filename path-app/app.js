const imageList = ["images/A.png", "images/B.png",
 "images/C.png", "images/D.png",
  "images/E.png","images/F.png",
  "images/G.png","images/H.png",
  "images/I.png","images/J.png",
  "images/K.png","images/L.png",
  "images/M.png","images/N.png",
  "images/O.png","images/P.png",
  "images/Q.png","images/R.png",
  "images/S.png","images/T.png",
  "images/U.png","images/V.png",
  "images/W.png","images/X.png",
  "images/Y.png","images/Z.png"]

var leftIndex = 0
var centerIndex = 0
var rightIndex = 0

var names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
var choiceMatrix = Array(imageList.length)
var totalChoices = 0
var nodeData = {"nodes":[], "links": []}
for (i = 0; i < imageList.length; i++){
	nodeData.nodes.push({"id": i, "name": names[i]})
	nodeData.links.push({"source": i,"target": 0})
	choiceMatrix[i] = Array(imageList.length).fill(0)
}
console.log(nodeData)

// color code from https://codenebula.io/javascript/frontend/dataviz/2019/04/18/automatically-generate-chart-colors-with-chart-js-d3s-color-scales/
const colorScale = d3.interpolateSpectral
const colorRangeInfo = {
	colorStart: 0,
	colorEnd: 1,
	useEndAsStart: false,
}

var nodeGroups = generateGroups(8)
console.log(nodeGroups)

function generateGroups(groups) {
	var nodeGroups = []
	for (i = 0; i < imageList.length; i++) {
		var groupNumber = Math.round(Math.random() * groups + 1)
		nodeGroups.push(groupNumber)
	}
	return nodeGroups
}

function updateBestPath(){
	nodeData.links = []
	for (i = 0; i < imageList.length; i++){
		var max = 0
		var max2 = 0
		var bestPath = i
		var bestPath2 = i
		for (j = 0; j < imageList.length; j++){
			if (choiceMatrix[i][j] > max2) {
				if (choiceMatrix[i][j] > max) {
					max2 = max
					bestPath2 = bestPath
					max = choiceMatrix[i][j]
					bestPath = j
				} else {
					max2 = choiceMatrix[i][j]
					bestPath2 = j
				}
			}
		}
		nodeData.links.push({"source": i,"target": bestPath, "value": choiceMatrix[i][bestPath]})
		nodeData.links.push({"source": i,"target": bestPath2, "value": choiceMatrix[i][bestPath2]})
	}
	console.log(nodeData)
}
    
/* Must use an interpolated color scale, which has a range of [0, 1] */
function interpolateColors(dataLength, colorScale, colorRangeInfo) {
	var { colorStart, colorEnd } = colorRangeInfo;
	var colorRange = colorEnd - colorStart;
	var intervalSize = colorRange / dataLength;
	var i, colorPoint;
	var colorArray = [];

	for (i = 0; i < dataLength; i++) {
	colorPoint = calculatePoint(i, intervalSize, colorRangeInfo);
	colorArray.push(colorScale(colorPoint));
	}

	return colorArray;
}
    

function calculatePoint(i, intervalSize, colorRangeInfo) {
	var { colorStart, colorEnd, useEndAsStart } = colorRangeInfo;
	return (useEndAsStart
	? (colorEnd - (i * intervalSize))
	: (colorStart + (i * intervalSize)));
}
    

function updateChordVisualization() {
	// code from https://www.d3-graph-gallery.com/graph/chord_interactive.html
	//deletes old svg
	document.getElementById("my_dataChordViz").innerHTML = ''
	// create the svg area
	var svg = d3.select("#my_dataChordViz")
	  .append("svg")
	    .attr("width", 850)
	    .attr("height", 850)
	  .append("g")
	    .attr("transform", "translate(425,425)")

	var colors = interpolateColors(imageList.length, colorScale, colorRangeInfo)

	// give this matrix to d3.chord(): it will calculates all the info we need to draw arc and ribbon
	var res = d3.chord()
	    .padAngle(0.05)
	    .sortSubgroups(d3.descending)
	    (choiceMatrix)

	// add the groups on the inner part of the circle
	svg
	  .datum(res)
	  .append("g")
	  .selectAll("g")
	  .data(function(d) { return d.groups; })
	  .enter()
	  .append("g")
	  .append("path")
	    .style("fill", "grey")
	    .style("stroke", function(d,i){return colors[i]})
	    .attr("d", d3.arc()
	      .innerRadius(400)
	      .outerRadius(420)
	    )

	// Add a tooltip div. Here I define the general feature of the tooltip: stuff that do not depend on the data point.
	// Its opacity is set to 0: we don't see it by default.
	var tooltip = d3.select("#my_dataChordViz")
	  .append("div")
	  .style("opacity", 0)
	  .attr("class", "tooltip")
	  .style("background-color", "white")
	  .style("border", "solid")
	  .style("border-width", "1px")
	  .style("border-radius", "5px")
	  .style("padding", "10px")

	// A function that change this tooltip when the user hover a point.
	// Its opacity is set to 1: we can now see it. Plus it set the text and position of tooltip depending on the datapoint (d)
	var showTooltip = function(d) {
	  tooltip
	    .style("opacity", 1)
	    .html("Source: " + names[d.source.index] + "<br>Target: " + names[d.target.index]
	    	+ "<br>Values: " + choiceMatrix[d.source.index][d.target.index] + " -> " + choiceMatrix[d.target.index][d.source.index])
	    .style("left", (d3.event.pageX + 15) + "px")
	    .style("top", (d3.event.pageY - 28) + "px")
	}


	// Add the links between groups
	svg
	  .datum(res)
	  .append("g")
	  .selectAll("path")
	  .data(function(d) { return d; })
	  .enter()
	  .append("path")
	    .attr("d", d3.ribbon()
	      .radius(380)
	    )
	    .style("fill", function(d){ return(colors[d.source.index]) })
	    .style("stroke", "black")
	  .on("mouseover", showTooltip )
}

function updateNodeVisualization() {
// Code from https://bl.ocks.org/d3indepth/c48022f55ebc76e6adafa77cf466da35

var width = 800, height = 800, radius = 50


var simulation = d3.forceSimulation(nodeData.nodes)
  .force('collide', d3.forceCollide().radius(20))
  .force('charge', d3.forceManyBody().strength(-50))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('link', d3.forceLink().links(nodeData.links))
  .on('tick', ticked);

function updateLinks() {
  var u = d3.select('.links')
    .selectAll('line')
    .data(nodeData.links)

  u.enter()
    .append('line')
    .merge(u)
    .attr('x1', function(d) {
      return d.source.x
    })
    .attr('y1', function(d) {
      return d.source.y
    })
    .attr('x2', function(d) {
      return d.target.x
    })
    .attr('y2', function(d) {
      return d.target.y
    })

  u.exit().remove()
}

function updateNodes() {
  u = d3.select('.nodes')
    .selectAll('text')
    .data(nodeData.nodes)

  u.enter()
    .append('text')
    .text(function(d) {
      return d.name
    })
    .merge(u)
    .attr('x', function(d) {
      return d.x = Math.max(radius, Math.min(width - radius, d.x))
    })
    .attr('y', function(d) {
      return d.y = Math.max(radius, Math.min(height - radius, d.y))
    })
    .attr('dy', function(d) {
      return 5
    })

  u.exit().remove()
}

function ticked() {
  updateLinks()
  updateNodes()
}
}

function randomStart() {
	leftIndex = Math.trunc(imageList.length* Math.random())
	centerIndex = Math.trunc(imageList.length* Math.random())
	rightIndex = Math.trunc(imageList.length* Math.random())
	while (leftIndex == rightIndex || leftIndex == centerIndex || centerIndex == rightIndex) {
		leftIndex = Math.trunc(imageList.length* Math.random())
		rightIndex = Math.trunc(imageList.length* Math.random())
	}
	document.getElementById("left-image").src = imageList[leftIndex]
	document.getElementById("center-image").src = imageList[centerIndex]
	document.getElementById("right-image").src = imageList[rightIndex]
}


function randomSides() {
	leftIndex = Math.trunc(imageList.length* Math.random())
	rightIndex = Math.trunc(imageList.length* Math.random())
	while (leftIndex == rightIndex || leftIndex == centerIndex || centerIndex == rightIndex) {
		leftIndex = Math.trunc(imageList.length* Math.random())
		rightIndex = Math.trunc(imageList.length* Math.random())
	}
	document.getElementById("left-image").src = imageList[leftIndex]
	document.getElementById("right-image").src = imageList[rightIndex]
}

function selectLeft() {
	choiceMatrix[centerIndex][leftIndex]++
	totalChoices++
	document.getElementById("center-image").src = document.getElementById("left-image").src
	centerIndex = leftIndex
	randomSides()
}

function selectRight() {
	choiceMatrix[centerIndex][rightIndex]++
	totalChoices++
	document.getElementById("center-image").src = document.getElementById("right-image").src
	centerIndex = rightIndex
	randomSides()
}

function selectSkip() {
	x = Math.trunc(Math.random()*100)
	if (x < 92) {
		if (Math.abs(nodeGroups[centerIndex] - nodeGroups[leftIndex]) < Math.abs(nodeGroups[centerIndex] - nodeGroups[rightIndex])) {
			selectLeft()
		} else {
			selectRight()
		}
	} else {
		if (x % 2 == 0) {
			selectLeft()
		} else {
			selectRight()
		}
	}

}

function updateText() {
	document.getElementById("text-output").innerHTML = "Displaying: " + totalChoices + " choices"
}

document.addEventListener("keydown", function(event) {
	if (event.code == 'KeyA' || event.code == 'ArrowLeft') {
		selectLeft()
	}
	if (event.code == 'KeyD' || event.code == 'ArrowRight') {
		selectRight()
	}
	if (event.code == 'KeyZ') {
		selectSkip()
	}
	if (event.code == 'KeyX'){
		for (i = 0; i < 100; i++){
			selectSkip()
		}
	}
	if (event.code == 'KeyC'){
		for (i = 0; i < 1000; i++){
			selectSkip()
		}
	}
	updateChordVisualization()
	updateBestPath()
	updateNodeVisualization()
	updateText()
})

document.addEventListener('DOMContentLoaded', () => {
	//start up
	randomStart()

})












