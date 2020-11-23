drop external table if exists imagenet_valid_data_out;
CREATE WRITABLE EXTERNAL TABLE imagenet_valid_data_out (LIKE imagenet_valid_data) 
   LOCATION ('gpfdist://worker0:8101/imagenet_valid_0.out',
    'gpfdist://worker1:8101/imagenet_valid_1.out',
    'gpfdist://worker2:8101/imagenet_valid_2.out',
    'gpfdist://worker3:8101/imagenet_valid_3.out',
    'gpfdist://worker4:8101/imagenet_valid_4.out',
    'gpfdist://worker5:8101/imagenet_valid_5.out',
    'gpfdist://worker6:8101/imagenet_valid_6.out',
    'gpfdist://worker7:8101/imagenet_valid_7.out'
   )
   FORMAT 'TEXT' ( DELIMITER '|' NULL ' ')
   DISTRIBUTED BY (id);
INSERT INTO imagenet_valid_data_out SELECT * FROM imagenet_valid_data;

drop external table if exists imagenet_train_data_out;
CREATE WRITABLE EXTERNAL TABLE imagenet_train_data_out (LIKE imagenet_train_data) 
   LOCATION ('gpfdist://worker0:8101/imagenet_train_0.out',
    'gpfdist://worker1:8101/imagenet_train_1.out',
    'gpfdist://worker2:8101/imagenet_train_2.out',
    'gpfdist://worker3:8101/imagenet_train_3.out',
    'gpfdist://worker4:8101/imagenet_train_4.out',
    'gpfdist://worker5:8101/imagenet_train_5.out',
    'gpfdist://worker6:8101/imagenet_train_6.out',
    'gpfdist://worker7:8101/imagenet_train_7.out'
   )
   FORMAT 'TEXT' ( DELIMITER '|' NULL ' ')
   DISTRIBUTED BY (id);
INSERT INTO imagenet_train_data_out SELECT * FROM imagenet_train_data;