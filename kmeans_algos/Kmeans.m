classdef Kmeans < handle
    
    properties (SetAccess = protected)
        K       % Number of centers
        lib     % Name of the lib
        lib_id  % id of the lib
        maxiter % Maximum number of iterations allowed
        points  % store the points to cluster (one per column)
    end
    
    methods
        %------------------------------------------------------------------        
        function obj = Kmeans(num_centers, library, maxiter)
            obj.K = num_centers;
            obj.lib = library;
            obj.maxiter = maxiter;
            
            if(strcmpi(library, 'vl'))
                obj.lib_id = 0;
            else
                if(strcmpi(library, 'vgg'))
                    obj.lib_id = 1;
                else
                    if(strcmpi(library, 'ml'))
                        obj.lib_id = 2;
                    else
                        if(strcmpi(library, 'mex'))
                            obj.lib_id = 3;
                        else
                            if(strcmpi(library, 'c'))
                                obj.lib_id = 4;
                            else                            
                                throw(MException('',['Unknown library for computing K-means: "' library '".\nPossible values are: "vl", "vgg", "ml", "mex" and "c".\n']));
                            end
                        end
                    end
                end
            end            
        end
               
        %------------------------------------------------------------------
        % prepare kmeans computation (one point per line)
        function n = prepare_kmeans(obj, points, file)
            global FILE_BUFFER_PATH;
            
            if nargin == 2 || (nargin == 3 && exist(file,'file') ~= 2) 
                n = 0;
                for i=1:size(points,1)
                    n = n + size(points{i}, 1);
                end

                switch obj.lib_id
                    case 0  % vlfeat
                        points = cat(1,points{:})';
                        m = max(max(points));
                        obj.points = uint8(255/m*points);
                    case {1, 3}  % vgg & mex
                        obj.points = cat(1,points{:})';
                    case 2  % matlab
                        obj.points = cat(1,points{:});
                    case 4  % cpp             	
                        file_in = fullfile(FILE_BUFFER_PATH,'input'); % if modified, modifiy also line 133 

                        i = 1;
                        while isempty(points{i})
                            i = i+1;
                        end
                        dimension = size(points{i}, 2);

                        % Save data
                        fid = fopen(file_in, 'w+');
                        fwrite(fid, dimension, 'int32');
                        fwrite(fid, n, 'int32');
                        for i=1:size(points,1)
                            fwrite(fid, points{i}', 'single');
                        end
                        fclose(fid);

                        obj.points = dimension;
                end
            end
        end
        
        %------------------------------------------------------------------
        % prepare kmeans computation (one point per line)
        function prepare_kmeans_fused(obj, points, file)
            global FILE_BUFFER_PATH;
            
            if nargin == 2 || (nargin == 3 && exist(file,'file') ~= 2)        
                switch obj.lib_id
                    case 0  % vlfeat
                        points = points';
                        m = max(max(points));
                        obj.points = uint8(255/m*points);
                    case {1, 3}  % vgg & mex
                        obj.points = points';
                    case 2  % matlab
                        obj.points = points;
                    case 4  % cpp             	
                        file_in = fullfile(FILE_BUFFER_PATH,'input'); % if modified, modifiy also line 133 

                        n = size(points, 1);
                        dimension = size(points, 2);

                        % Save data
                        fid = fopen(file_in, 'w+');
                        fwrite(fid, dimension, 'int32');
                        fwrite(fid, n, 'int32');
                        fwrite(fid, points', 'single');
                        fclose(fid);

                        obj.points = dimension;
                end
            end
        end        
        
        %------------------------------------------------------------------
        function centers = do_kmeans(obj, file)
            if nargin >= 2 && exist(file,'file') == 2
                load(file,'centers');
                if exist('centers','var') ~= 1
                    load(file,'c');
                    if exist('c','var') == 1
                        centers = c;
                        save(file, 'centers');
                    end
                end
            end
            if exist('centers','var') ~= 1
                switch obj.lib_id
                case 0  % vlfeat
                    centers = obj.vlfeat(obj.K, obj.maxiter);
                case 1  % vgg
                    centers = obj.vgg(obj.K, obj.maxiter);
                case 2  % matlab
                    centers = obj.matlab(obj.K, obj.maxiter);
                case 3  % mex
                    centers = obj.mex(obj.K, obj.maxiter);
                case 4  % cpp             	
                    centers = obj.cpp(obj.K, obj.maxiter);
                end
                if nargin >= 2
                    save(file, 'centers');
                end
            end
            
            obj.points = [];
        end    
        
        %------------------------------------------------------------------
        function l = get_lib(obj)
            l = obj.lib;
        end
    
        %------------------------------------------------------------------
        function centers = vlfeat(obj, K, maxiter)
            centers = vl_ikmeans(obj.points, K, 'MaxIters', maxiter);
            centers = m/255*double(centers');            
        end
        
        %------------------------------------------------------------------
        function centers = vgg(obj, K, maxiter)
            centers = vgg_kmeans(obj.points, K, maxiter)';            
        end
        
        %------------------------------------------------------------------
        function centers = matlab(obj, K, maxiter)
            [id centers] = kmeans(obj.points, K, 'emptyaction', 'singleton','onlinephase','off');
        end
        
        %------------------------------------------------------------------
        function centers = mex(obj, K, maxiter)
            centers = kmeans_mex(obj.points, K, maxiter);
            centers = centers';            
        end
        
        %------------------------------------------------------------------
        function centers = cpp(obj, K, maxiter)
            global FILE_BUFFER_PATH LIB_DIR;
            
            file_in = fullfile(FILE_BUFFER_PATH,'input');
            file_out = fullfile(FILE_BUFFER_PATH,'output');
                        
            % Do kmeans
            cmd = fullfile(LIB_DIR, 'kmeans', sprintf('kmeans_cpp %s %d %d %s', file_in, K, maxiter, file_out));
            system(cmd);
            
            % Load data
            fid = fopen(file_out, 'r');
            centers = fread(fid, K*obj.points, 'single');   % hack: obj.points is the dimension of the points
            fclose(fid);
            centers = reshape(centers, obj.points, K)';     % hack: obj.points is the dimension of the points
        end
        %------------------------------------------------------------------
    end    
end

