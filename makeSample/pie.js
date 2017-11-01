var NUM_FILE = 100;
var chartWidth = 500, chartHeight = 500;


var fs = require('fs');
var d3 = require('d3');
var jsdom = require('jsdom');
var svg_to_png = require('svg-to-png');
var path = require('path')


var color = d3.scale.category10();

module.exports = function( pieData, outputLocation ){

    for (var i=0; i< NUM_FILE; i++) {
      
	(function (i) {

	    var r = Math.random()

	    pieData = [r,1-r];
	    outputLocation = i + '.svg';

	    var size = Math.floor(Math.random() * chartWidth /2)

	    var arc = d3.svg.arc()
		.outerRadius(size)
		.innerRadius(0);

	    jsdom.env({
		html:'',
		features:{ QuerySelector:true }, //you need query selector for D3 to work
		done:function(errors, window){
		    window.d3 = d3.select(window.document); //get d3 into the dom

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
			.attr('transform','translate(' + chartWidth/2 + ',' + chartWidth/2 + ')');

		    svg.selectAll('.arc')
			.data( d3.layout.pie()(pieData) )
			.enter()
			.append('path')
			.attr({
			    'class':'arc',
			    'd':arc,
			    'fill':function(d,i){
				c = Math.floor((Math.random() * 10) + 1);
				return color(c);
			    },
			    'stroke':'#fff'
			});

		    //write out the children of the container div
		    fs.writeFileSync(outputLocation, window.d3.select('.container').html()) //using sync to keep the code simple
		    //svg_to_png.convert(path.join(__dirname, outputLocation), path.join(__dirname, "bitmap")).then(function() {

		    //});

		    console.log(outputLocation)
		}
	    });

	})(i);

    }
}

if (require.main === module) {
    module.exports();
}
