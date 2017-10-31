// type argument like node ~.js <value>
var NUM_FILE = parseInt(process.argv[2]);
var chartWidth = 500, chartHeight = 500;


var fs = require('fs');
var d3 = require('d3');
var jsdom = require('jsdom');
var svg_to_png = require('svg-to-png');
var path = require('path')


var color = d3.scale.category10();

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

        r = (Math.random()+0.2)*100; // radius : 20 ~ 120
        circleData = [r];
		    outputLocation =  i + '.svg';

        //set coordinate of the circle, (x,y)
        /*
        rand_X = 0;
        while( rand_X < 1e-2*r ) { rand_X = Math.random() + 0.2; }
        rand_Y = 0;
        while( rand_Y < 1e-2*r ) { rand_Y = Math.random() + 0.2; }
        */
        rand_X = Math.random();
        rand_Y = Math.random();
		    // var coord_X = Math.floor((rand_X - 0.2) * chartWidth - r);
        // var coord_Y = Math.floor((rand_Y - 0.2) * chartWidth - r);
        var coord_X = Math.floor((rand_X) * chartWidth);
        var coord_Y = Math.floor((rand_Y) * chartWidth);

		    //do yr normal d3 stuff
		    var svg = window.d3.select('body')
			.append('div').attr('class','container') //make a container div to ease the saving process
			.append('svg')
			.attr({
			    xmlns:'http://www.w3.org/2000/svg',
			    width:chartWidth,
			    height:chartHeight
			})
			.append('g')
			.attr('transform','translate(' + coord_X + ',' + coord_Y + ')'); //set place where circle is.

		    svg.selectAll('.circle')
      .data(circleData)
      .enter()
      .append("circle")
      .attr({
        'class':'circle',
        // "cx": 50, //or, we can also set a coordinate of circle like this.
        // "cy": 50,
        "r": function(d){ return d }
      })
      .style("fill", function() {
        return "hsl(" + Math.random() * 360 + ",100%,50%)"; // set random color
      });

		    //write out the children of the container div
		    fs.writeFileSync(path.join(__dirname, 'svg', outputLocation), window.d3.select('.container').html()) //using sync to keep the code simple
		    svg_to_png.convert(path.join(__dirname, 'svg', outputLocation), path.join(__dirname, "bitmap")).then(function() {

		    });
		    console.log(outputLocation)
		}
	    });


	})(i);

    }
}

if (require.main === module) {
    module.exports();
}

