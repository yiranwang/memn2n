define([], function() {

    return '<div class="col-sm-6 col-md-4">' +
            '<div class="thumbnail">' +
              '<div class="caption">' +
                '<div class="well"><%= story %></div>' +
                '<h4><%= question %></h4>' +
                '<span class="label label-default"><%= answer %></span>' +
                '<p><a href="#" class="btn btn-primary" role="button">Select</a></p>' +
              '</div>' +
            '</div>' +
          '</div>'
});


