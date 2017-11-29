// type argument like node ~.js <value>
var NUM_FILE = parseInt(process.argv[2]);
var chartWidth = 500, chartHeight = 500;

var fs = require('fs');
var d3 = require('d3');
var jsdom = require('jsdom');
var svg_to_png = require('svg-to-png');
var path = require('path')

var dir = path.join(__dirname, 'svg');
if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
}
module.exports = function( pieData, outputLocation ){

  for (var i=(NUM_FILE * 100); i< (NUM_FILE+1) * 100; i++) {
	   (function (i) {
	      jsdom.env({
      		html:'',
      		features:{ QuerySelector:true }, //you need query selector for D3 to work
      		done:function(errors, window){
      		    window.d3 = d3.select(window.document); //get d3 into the dom
          	  outputLocation =  i + '.svg';

              var margin = {top: 30, right: 40, bottom: 70, left: 60},
                  width = chartWidth - margin.left - margin.right,
                  height = chartHeight - margin.top - margin.bottom;

              var x = d3.scale.ordinal().rangeRoundBands([0, width], .05);
              var y = d3.scale.linear().range([height, 0]);

              data = []
              //1~3
              var rand_temp = Math.floor(Math.random()*3) + 1
              var rand = Math.floor(Math.random() * 10) + 3
              for(var j=0 ; j<rand ; ++j){
                data[j] = {'letter' : makeid()};
                data[j]['frequency'] = Math.floor(Math.random() * Math.pow(10,rand_temp) * rand_temp);
              }
              //30 * (1 ~ 12)
              var color = Math.floor(Math.random()*13) * 30;

              x.domain(data.map(function(d) { return d.letter; }));
              y.domain([0, d3.max(data, function(d) { return d.frequency; })]);

              var xAxis = d3.svg.axis()
                  .scale(x)
                  .orient("bottom")
                  .tickSize(0,2)

              var yAxis = d3.svg.axis()
                  .scale(y)
                  .orient("left")
                  .tickSize(0,2)
                  .ticks(10)

              //do yr normal d3 stuff
      		    var svg = window.d3.select('body')
        			.append('div').attr('class','container') //make a container div to ease the saving process
        			.append('svg')
              .attr({
                  xmlns:'http://www.w3.org/2000/svg',
                  width:chartWidth,
                  height:chartHeight
              })
              .append("g")
              .attr("transform",
                    "translate(" + margin.left + "," + margin.top + ")");

              svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(xAxis)
                  .selectAll("text")
                    .style("text-anchor", "end")
                    .attr("dx", "-.8em")
                    .attr("dy", "0em")
                    .attr("transform", "rotate(-90)" );

                svg.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                  .append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", ".71em")
                    .style("text-anchor", "end")
                    // .text("Value ($)");

                svg.selectAll("bar")
                    .data(data)
                  .enter().append("rect")
                    .style("fill", "hsl(" + color + ",100%,50%)")
                    .attr("x", function(d) { return x(d.letter); })
                    .attr("width", x.rangeBand())
                    .attr("y", function(d) { return y(d.frequency); })
                    .attr("height", function(d) { return height - y(d.frequency); });

      		    //write out the children of the container div
      		    fs.writeFileSync(path.join(__dirname, 'svg', outputLocation), window.d3.select('.container').html()) //using sync to keep the code simple
      		    svg_to_png.convert(path.join(__dirname, 'svg', outputLocation), path.join(__dirname, "bitmap")).then(function() {});
      		    console.log(outputLocation)
		      }
	    })
  	})(i)
  }
}

function makeid() {
  var text = "";
  var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  for (var a = 0; a < 5; a++)
    text += possible.charAt(Math.floor(Math.random() * possible.length));

  return text;
}

if (require.main === module) {
    module.exports();
}
