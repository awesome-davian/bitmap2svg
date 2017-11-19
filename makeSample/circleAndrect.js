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

        var circle_id = 'circle';
	var rect_id = 'rect';
        var circle_object = {
          'class': 'circle',
          "r": function(d){ return d }
        };
	var rect_object = {
          'class': 'rect',
          "width": function(d){ return d },
          "height": function(d){ return d }
        };
        r = (Math.random()+0.2)*100; // radius : 20 ~ 120
        sizeData = [r];
		    outputLocation =  i + '.svg';

        
        //set coordinate of the circle, (x,y)
        circle_rand_X = Math.random();
        circle_rand_Y = Math.random();
	rect_rand_X = Math.random();
	rect_rand_Y = Math.random();	
	    
	// var coord_X = Math.floor((rand_X - 0.2) * chartWidth - r);
        // var coord_Y = Math.floor((rand_Y - 0.2) * chartWidth - r);
        var circle_coord_X = Math.floor((circle_rand_X) * chartWidth);
        var circle_coord_Y = Math.floor((circle_rand_Y) * chartHeight);
	var rect_coord_X = Math.floor((rect_rand_X) * chartWidth);
	var rect_coord_Y = Math.floor((rect_rand_Y) * chartHeight)	

		    //do yr normal d3 stuff
		    var svg = window.d3.select('body')
			.append('div').attr('class','container') //make a container div to ease the saving process
			.append('svg')
			.attr({
			    xmlns:'http://www.w3.org/2000/svg',
			    width:chartWidth,
			    height:chartHeight
			})
/*			.append('circle')
			.attr('transform','translate(' + circle_coord_X + ',' + circle_coord_Y + ')') //set place where circle is.
			.append('rect')
			.attr('transform','translate(' + rect_coord_X + ',' + rect_coord_Y + ')')
*/	

		    svg.selectAll('circle')
      		    	
			.append('circle')
			.data(sizeData)
      			.enter()
      		    	.append(circle_id)
			.attr('transform','translate(' + circle_coord_X + ',' + circle_coord_Y + ')') //set place where circle is.
			.attr(circle_object)
      		    	.style("fill", function() {
        			return "hsl(" + Math.random() * 360 + ",100%,50%)"; // set random color
      		    });
		    
		    svg.selectAll('rect')

		    	.append('rect')	
			.data(sizeData)
			.enter()
		    	.append(rect_id)
			.attr('transform','translate(' + rect_coord_X + ',' + rect_coord_Y + ')')
                    	.attr(rect_object)
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

